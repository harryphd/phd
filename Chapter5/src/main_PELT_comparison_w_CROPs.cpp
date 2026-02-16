#include <iostream>
#include <set>
#include <tuple>
#include <cmath>
#include <chrono>
#include <fstream>
#include <vector>
#include <random>
#include <map>
#include "EdPeltChangePointDetector.h"
#include "GPDEdPeltChangePointDetectorCorrected2.h"

#include <thread>
#include <mutex>
#include <functional>
#include <sstream>

// Global mutex for thread-safe output to shared resources
std::mutex output_mutex;
std::mutex crops_cache_mutex;  // Global or local to crops_impl

using namespace std;
using namespace Eigen;
using namespace std::chrono;

// Struct to store the cached result
struct CacheEntry {
    double Qm; // Total cost
    int m;     // Number of change points
};

// Struct to store the results of change point detection
struct ChangePointResultsOutput {
    vector<vector<int>> ed_pelt_changepoints;
    vector<vector<int>> gpd_ed_pelt_changepoints;
};

// Forward declaration of print_progress
void print_progress(int current, int total, double elapsed_time);

// Function to generate a unique seed for each thread/experiment
unsigned generate_seed(int experiment_id) {
    std::random_device rd;  // Random device to generate non-deterministic seed
    auto timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();  // Current timestamp

    // Combine random device output, experiment ID, and timestamp using std::hash
    std::size_t hash_value = std::hash<std::size_t>{}(rd() ^ experiment_id ^ timestamp);

    return static_cast<unsigned>(hash_value);  // Cast hash value to unsigned int for seed
}

// CROPs implementation with thread-safe cache access
template<typename Detector>
set<double> crops_impl(Detector& detector, double (*f)(Detector&, const double*, int, double, int, int&), double beta_min, double beta_max, const double* data, int n, int max_iter = 20) {
    // Initialize beta_star with the initial interval [beta_min, beta_max]
    set<tuple<double, double>> beta_star = { make_tuple(beta_min, beta_max) };
    set<double> res;
    map<double, CacheEntry> cache; // Cache to store computed Qm and m values for each beta
    int total_iterations = max_iter;
    int current_iteration = 0;
    bool completed_early = false;
    
    while (!beta_star.empty() && max_iter > 0) {
        auto start_time = high_resolution_clock::now(); // Start timing

        --max_iter;
        ++current_iteration;
        
        // Get the last tuple (beta_0, beta_1) from beta_star
        auto it = beta_star.rbegin();
        double beta_0 = get<0>(*it);
        double beta_1 = get<1>(*it);
        beta_star.erase(prev(it.base()));

        // Lock the mutex to ensure thread-safe access to the cache
        {
            std::lock_guard<std::mutex> lock(crops_cache_mutex);

            // Check if beta_0 and beta_1 are already in the cache
            if (cache.find(beta_0) == cache.end()) {
                // If not in cache, compute and store in cache
                int out_cpts_count_0 = 0;
                double Qm_0 = f(detector, data, n, beta_0, 300, out_cpts_count_0);
                cache[beta_0] = {Qm_0, out_cpts_count_0};
            }
            if (cache.find(beta_1) == cache.end()) {
                // If not in cache, compute and store in cache
                int out_cpts_count_1 = 0;
                double Qm_1 = f(detector, data, n, beta_1, 300, out_cpts_count_1);
                cache[beta_1] = {Qm_1, out_cpts_count_1};
            }
        }

        // Retrieve Qm and m values from the cache (outside the lock)
        double Qm_0 = cache[beta_0].Qm;
        int m_0 = cache[beta_0].m;
        double Qm_1 = cache[beta_1].Qm;
        int m_1 = cache[beta_1].m;

        // Calculate beta_int
        if (m_0 > m_1 + 1) {
            double beta_int = (Qm_1 - Qm_0) / static_cast<double>(m_0 - m_1);

            // Lock mutex again for cache update
            {
                std::lock_guard<std::mutex> lock(crops_cache_mutex);

                // Check if beta_int is in cache
                if (cache.find(beta_int) == cache.end()) {
                    int out_cpts_count_int = 0;
                    double Qm_int = f(detector, data, n, beta_int, 300, out_cpts_count_int);
                    cache[beta_int] = {Qm_int, out_cpts_count_int};
                }
            }

            int m_int = cache[beta_int].m;

            if (m_int != m_1) {
                beta_star.insert(make_tuple(beta_int, beta_1));
                beta_star.insert(make_tuple(beta_0, beta_int));
            }
        }
        
        // Update res with beta_0
        res.insert(beta_0);

        if (beta_star.empty()) {
            // If the loop will exit early, note it
            completed_early = true;
            break;
        }

        auto end_time = high_resolution_clock::now(); // End timing
        duration<double> iteration_time = duration_cast<duration<double>>(end_time - start_time);
    }

    cout << endl << "Done" << endl;
    
    return res;
}

// Function to generate a time series
vector<double> generate_series(int num_points, int change, double phi1, double theta1, double theta2, double df_high, double df_low, double norm_param, unsigned seed) {
    mt19937 gen(seed);
    student_t_distribution<double> dist_high(df_high);
    student_t_distribution<double> dist_low(df_low);
    normal_distribution<double> norm_dist(0, 1);

    vector<double> epsilon(num_points);
    for (int i = 0; i < change; ++i) {
        epsilon[i] = dist_high(gen);
    }
    for (int i = change; i < num_points; ++i) {
        epsilon[i] = dist_low(gen);
    }

    vector<double> Y(num_points, 0);
    for (int t = 1; t < num_points; ++t) {
        Y[t] = phi1 * Y[t - 1] + epsilon[t] + theta1 * epsilon[t - 1] + theta2 * epsilon[t - 2] + norm_param * norm_dist(gen);
    }

    return Y;
}

// Function to print progress
void print_progress(int current, int total, double elapsed_time) {
    int barWidth = 50;
    float progress = static_cast<float>(current) / static_cast<float>(total);
    int pos = static_cast<int>(barWidth * progress);

    cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) cout << "=";
        else if (i == pos) cout << ">";
        else cout << " ";
    }
    cout << "] " << int(progress * 100.0) << "% (" << elapsed_time << "s per iteration)\r";
    cout.flush();
}

// Example usage with EdPeltChangePointDetector
double evaluate_change_points(EdPeltChangePointDetector& detector, const double* data, int n, double penalty_n, int min_distance, int& out_cpts_count) {
    auto result = detector.get_change_point_indexes(data, n, penalty_n, min_distance, out_cpts_count);
    return result.total_cost;
}

// Example usage with GPDEdPeltChangePointDetector
double evaluate_change_points_gpd(GPDEdPeltChangePointDetector& detector, const double* data, int n, double penalty_n, int min_distance, int& out_cpts_count) {
    auto result = detector.get_change_point_indexes(data, n, penalty_n, min_distance, out_cpts_count);
    return result.total_cost;
}

// Modified run_experiment function to handle a range of experiments with unique seeding
void run_experiment_range(int start, int end, int num_points, int change, double phi1, double theta1, double theta2, double df_high, double df_low, double norm_param, GPDEdPeltChangePointDetector& gpd_ed_detector, ChangePointResultsOutput& results_output, unsigned start_seed) {
    for (int i = start; i < end; ++i) {
        auto start_time = high_resolution_clock::now(); // Start timing

        // Use the seed as the experiment number + start_seed
        unsigned seed = start_seed + i;
        auto series = generate_series(num_points, change, phi1, theta1, theta2, df_high, df_low, norm_param, seed);

        double beta_min_ed = 0.2 * std::log(static_cast<double>(num_points)); //  minimum penalty
        double beta_max_ed = 50.0 * std::log(static_cast<double>(num_points)); //  maximum penalty

        // Find the best penalty using CROPs
        EdPeltChangePointDetector ed_detector;
        auto ed_crops_results = crops_impl(ed_detector, evaluate_change_points, beta_min_ed, beta_max_ed, series.data(), num_points);
        double best_penalty_ed = *ed_crops_results.rbegin();  // The last penalty value

        // ED-PELT with the best penalty
        int out_cpts_count = 0; // Variable to capture the number of change points
        ChangePointResult result = ed_detector.get_change_point_indexes(series.data(), series.size(), best_penalty_ed + 0.01, 300, out_cpts_count);
        int* change_point_inds = result.change_points;
        vector<int> cpts_vector(change_point_inds, change_point_inds + out_cpts_count);
        delete[] change_point_inds; // Clean up dynamically allocated array

        // // Find the best penalty using CROPs for GPD-ED-PELT
        double beta_min_gpep = 1.0 * std::log(static_cast<double>(num_points)); //  minimum penalty
        double beta_max_gpep = 500.0 * std::log(static_cast<double>(num_points)); //  maximum penalty
        auto gpd_crops_results = crops_impl(gpd_ed_detector, evaluate_change_points_gpd, beta_min_gpep, beta_max_gpep, series.data(), num_points);
        double best_penalty_gpd = *gpd_crops_results.rbegin();  // The last penalty value

        // GPD-ED-PELT with the best penalty
        int gpd_out_cpts_count = 0;
        ChangePointResultGPD gpd_result = gpd_ed_detector.get_change_point_indexes(series.data(), series.size(), best_penalty_gpd + 0.01, 300, gpd_out_cpts_count);
        int* gpd_change_point_inds = gpd_result.change_points;
        vector<int> gpd_cpts_vector(gpd_change_point_inds, gpd_change_point_inds + gpd_out_cpts_count);
        delete[] gpd_change_point_inds; // Clean up dynamically allocated array

        // Update shared results_output with a mutex to prevent data races
        {
            std::lock_guard<std::mutex> lock(output_mutex);
            results_output.ed_pelt_changepoints.push_back(cpts_vector);
            results_output.gpd_ed_pelt_changepoints.push_back(gpd_cpts_vector);
        }

        auto end_time = high_resolution_clock::now(); // End timing
        duration<double> iteration_time = duration_cast<duration<double>>(end_time - start_time);
        print_progress(i + 1 - start, end - start, iteration_time.count());
    }
}

// Function to save results to a file
void save_to_txt(const string& filename, const vector<vector<int>>& data) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return;
    }

    for (const auto& item : data) {
        for (int cp : item) {
            file << cp << " ";
        }
        file << endl;
    }
    file.close();
}

// Function to save results to a CSV file
void save_to_csv(const string& filename, const vector<vector<int>>& ed_pelt_data, const vector<vector<int>>& gpd_ed_pelt_data) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Failed to open file: " << filename << endl;
        return;
    }

    // Assuming both ed_pelt_data and gpd_ed_pelt_data have the same size
    for (size_t i = 0; i < ed_pelt_data.size(); ++i) {
        // Write ED-PELT change points
        for (size_t j = 0; j < ed_pelt_data[i].size(); ++j) {
            file << ed_pelt_data[i][j];
            if (j != ed_pelt_data[i].size() - 1) file << " "; // Separate change points by space
        }

        file << ","; // Separate ED-PELT and GPD-ED-PELT change points

        // Write GPD-ED-PELT change points
        for (size_t j = 0; j < gpd_ed_pelt_data[i].size(); ++j) {
            file << gpd_ed_pelt_data[i][j];
            if (j != gpd_ed_pelt_data[i].size() - 1) file << " ";
        }

        file << endl; // Newline after each experiment
    }
    
    file.close();
}


// Main function modified for threading
int main() {
    int num_points = 5000;
    double df = 1.4;
    int true_changepoint = 2500;
    int num_experiments = 500;
    int num_threads = std::thread::hardware_concurrency(); // Get number of available threads

    unsigned start_seed = 1; // Specify your starting seed here

    // Create the detector instance here so it persists through the experiment
    GPDEdPeltChangePointDetector gpd_ed_detector;

    // Output container
    ChangePointResultsOutput results_output;

    // Launch multiple threads to perform the experiments
    vector<thread> threads;
    int experiments_per_thread = num_experiments / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        int start = t * experiments_per_thread;
        int end = (t == num_threads - 1) ? num_experiments : start + experiments_per_thread;

        threads.emplace_back(run_experiment_range, start, end, num_points, true_changepoint, 0.0, 0.0, 0.0, df, 1.2, 0, std::ref(gpd_ed_detector), std::ref(results_output), start_seed);
    }

    // Join threads
    for (auto& th : threads) {
        th.join();
    }

    // Save results to a CSV file
    save_to_csv("final_results_95.csv", results_output.ed_pelt_changepoints, results_output.gpd_ed_pelt_changepoints);

    return 0;
}
