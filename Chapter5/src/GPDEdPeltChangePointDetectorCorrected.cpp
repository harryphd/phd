#include "GPDEdPeltChangePointDetectorCorrected2.h"
#include <iostream>
#include <numeric>
#include <cmath>
#include <algorithm>

// Include Eigen namespace
#include <Eigen/Dense>
using Eigen::MatrixXi;

// Define thread-local gpd_cache
thread_local std::map<std::vector<double>, std::vector<double>> GPDEdPeltChangePointDetector::gpd_cache;

// Constructor
GPDEdPeltChangePointDetector::GPDEdPeltChangePointDetector() : gpd_fit_count(0) {}

// Destructor
GPDEdPeltChangePointDetector::~GPDEdPeltChangePointDetector() {}

// Utility function to find the index of the minimum element in an array
int GPDEdPeltChangePointDetector::which_min(const double* values, int size) {
    int min_index = 0;
    for (int i = 1; i < size; ++i) {
        if (values[i] < values[min_index]) {
            min_index = i;
        }
    }
    return min_index;
}

ChangePointResultGPD GPDEdPeltChangePointDetector::get_change_point_indexes(const double* data, int n, double penalty_n, int min_distance, int& out_cpts_count) {
    if (n <= 2 || min_distance < 1 || min_distance > n) {
        throw std::invalid_argument("Invalid input size or minimum distance.");
    }

    double penalty = penalty_n;
    int k = std::min(n, static_cast<int>(std::ceil(4 * std::log(static_cast<double>(n)))));
    PartialSumsResult partial_sums_result = get_partial_sums(data, n, k);
    Eigen::MatrixXi partial_sums = partial_sums_result.partial_sums;
    std::vector<double> thresholds = partial_sums_result.thresholds;
    std::vector<double> p_values = partial_sums_result.p_values;  // Extract p_values from partial_sums_result

    // Calculate the 95th percentile
    std::vector<double> sorted_data(data, data + n);
    std::sort(sorted_data.begin(), sorted_data.end());
    double u = sorted_data[static_cast<int>((n - 1) * 0.959)];

    double* best_cost = new double[n + 1]();
    int* previous_change_point_index = new int[n + 1]();
    int* previous_taus = new int[n + 1]();
    int previous_taus_size = 0;

    best_cost[0] = -penalty;
    previous_taus[previous_taus_size++] = 0;

    for (int current_tau = min_distance; current_tau < 2 * min_distance; ++current_tau) {
        best_cost[current_tau] = get_segment_cost(partial_sums, thresholds, p_values, 0, current_tau, k, n, data, u);  // Pass p_values here
    }

    previous_taus[previous_taus_size++] = min_distance;
    double* cost_for_previous_tau = new double[n];

    for (int current_tau = 2 * min_distance; current_tau <= n; ++current_tau) {

        for (int i = 0; i < previous_taus_size; ++i) {
            int previous_tau = previous_taus[i];
            cost_for_previous_tau[i] = best_cost[previous_tau] +
                get_segment_cost(partial_sums, thresholds, p_values, previous_tau, current_tau, k, n, data, u) + penalty;  // Pass p_values here
        }

        int best_previous_tau_index = which_min(cost_for_previous_tau, previous_taus_size);
        best_cost[current_tau] = cost_for_previous_tau[best_previous_tau_index];
        previous_change_point_index[current_tau] = previous_taus[best_previous_tau_index];

        double current_best_cost = best_cost[current_tau];
        int new_size = 0;
        for (int i = 0; i < previous_taus_size; ++i) {
            int tau = previous_taus[i];
            if (best_cost[tau] + get_segment_cost(partial_sums, thresholds, p_values, tau, current_tau, k, n, data, u) + penalty < current_best_cost + penalty) {
                previous_taus[new_size++] = tau;
            }
        }
        previous_taus[new_size++] = current_tau - (min_distance - 1);
        previous_taus_size = new_size;
    }

    int* change_point_indexes = new int[n];
    int change_point_count = 0;
    double total_cost = 0.0;

    for (int current_index = previous_change_point_index[n]; current_index != 0; current_index = previous_change_point_index[current_index]) {
        change_point_indexes[change_point_count++] = current_index - 1;
    }
    std::reverse(change_point_indexes, change_point_indexes + change_point_count);
    out_cpts_count = change_point_count;

    // Calculate the total cost of all segments with the final changepoints
    int previous_tau = 0;
    for (int i = 0; i < change_point_count; ++i) {
        int current_tau = change_point_indexes[i] + 1;
        total_cost += get_segment_cost(partial_sums, thresholds, p_values, previous_tau, current_tau, k, n, data, u);  // Pass p_values here
        previous_tau = current_tau;
    }
    // Add the cost of the last segment
    total_cost += get_segment_cost(partial_sums, thresholds, p_values, previous_tau, n, k, n, data, u);  // Pass p_values here

    // Prepare the result to return
    ChangePointResultGPD result;
    result.change_points = change_point_indexes;
    result.total_cost = total_cost;

    // Cleanup
    delete[] best_cost;
    delete[] previous_change_point_index;
    delete[] previous_taus;
    delete[] cost_for_previous_tau;

    return result;
}


PartialSumsResult GPDEdPeltChangePointDetector::get_partial_sums(const double* data, int n, int k) {
    PartialSumsResult result;
    result.partial_sums = Eigen::MatrixXi::Zero(k, n + 1);
    result.thresholds.resize(k);

    std::vector<double> sorted_data(data, data + n);
    std::sort(sorted_data.begin(), sorted_data.end());

    double t_95 = sorted_data[static_cast<int>((n - 1) * 0.959)];  // Calculate the 95th percentile

    std::vector<double> p_values(k);  // This will store the p-values for each threshold

    for (int i = 0; i < k; ++i) {
        double z = -1 + (2 * i + 1.0) / k;
        double p = 1.0 / (1 + std::pow(2 * n - 1, -z));
        // std::cout << "pvalue" << p <<"\n";
        // pvalue 0.932861
        // pvalue 0.959215
        // pvalue 0.975496
        // pvalue 0.985377

        p_values[i] = p;  // Store the p-values
        double t = sorted_data[static_cast<int>((n - 1) * p)];
        result.thresholds[i] = t;

        for (int tau = 1; tau <= n; ++tau) {
            double y_j = data[tau - 1];

            if (t > t_95 && y_j > t_95) {
                // For thresholds above t_95, data points above t_95 contribute zero
                result.partial_sums(i, tau) = result.partial_sums(i, tau - 1);
            } else {
                result.partial_sums(i, tau) = result.partial_sums(i, tau - 1) + (y_j < t ? 2 : (y_j == t ? 1 : 0));
            }
        }
    }

    result.p_values = p_values;  // Store the p-values in the result
    return result;
}


double GPDEdPeltChangePointDetector::get_segment_cost(const Eigen::MatrixXi& partial_sums, const std::vector<double>& thresholds, const std::vector<double>& p_values, int tau1, int tau2, int k, int n, const double* data, double u) {
    double total_sum = 0.0;
    int nseg = tau2 - tau1;  // Total number of points in the segment

    // Compute F_star_u
    int count_below_u = 0;
    int count_equal_u = 0;
    std::vector<double> excesses;

    for (int j = tau1; j < tau2; ++j) {
        if (data[j] < u) {
            count_below_u++;
        } else if (data[j] == u) {
            count_equal_u++;
        } else {
            excesses.push_back(data[j] - u);
        }
    }

    double F_star_u = (count_below_u + 0.5 * count_equal_u) / nseg;

    // Fit GPD if there are excesses
    std::vector<double> gpd_params;
    if (!excesses.empty()) {
        std::vector<double> excesses_sorted = excesses; // Copy and sort for caching
        std::sort(excesses_sorted.begin(), excesses_sorted.end());

        // Check if we have already cached the GPD parameters for these excesses
        auto it = gpd_cache.find(excesses_sorted);
        if (it == gpd_cache.end()) {
            // Fit GPD to all the excesses and cache the result
            fit_gpd(excesses, gpd_params);
            gpd_cache[excesses_sorted] = gpd_params;
        } else {
            gpd_params = it->second;  // Retrieve from cache
        }
    }

    // Now loop over thresholds
    for (int i = 0; i < k; ++i) {
        double t_k = thresholds[i];

        double fit = 0.0;
        if (t_k <= u) {
            // Use partial sums
            int actual_sum = partial_sums(i, tau2) - partial_sums(i, tau1);
            fit = actual_sum * 0.5 / nseg;

            // std::cout << "Threshold " << i << ": fit = " << fit << std::endl; // Debugging line

            // std::cout << "Fit (below thresh): " << fit;


        } else {
            // t_k > u
            if (!excesses.empty()) {

                // std::cout << "GPD Parameters: ";
                // for (size_t idx = 0; idx < gpd_params.size(); ++idx) {
                //     std::cout << "Param[" << idx << "] = " << gpd_params[idx] << " ";
                // }
                // std::cout << "\n";

                // std::cout << "Excesses used for GPD fitting (" << excesses.size() << " values): ";
                // int limit = std::min(static_cast<int>(excesses.size()), 10);
                // for (int idx = 0; idx < limit; ++idx) {
                //     std::cout << excesses[idx] << " ";
                // }
                // if (excesses.size() > 10) {
                //     std::cout << "... (" << excesses.size() - 10 << " more)";
                // }
                // std::cout << "\n";

                double excess = t_k - u;

                double gpd_cdf_val;
                if (excess < 0) {
                    // std::cerr << "Error: Negative excess encountered (t_k - u = " << excess << "). Setting gpd_cdf_val to 0.\n";
                    gpd_cdf_val = 0.0;
                } else {
                    gpd_cdf_val = gpd_cdf(excess, gpd_params);  // GPD CDF at t_k - u
                }

                // Print gpd_cdf_val
                // std::cout << "GPD CDF Value for excess (" << excess << "): " << gpd_cdf_val << "\n";

                fit = F_star_u + (1 - F_star_u) * gpd_cdf_val;
                // std::cout << "Threshold " << i << ": fit = " << fit << std::endl; // Debugging line

                // std::cout << "Fit (above threshold): " << fit << "\n";
                // std::cout << "Fit (at threshold F_star_u): " << F_star_u << "\n";

                if (std::isnan(gpd_cdf_val)) {
                    // std::cerr << "Error: GPD CDF returned NaN for excess (" << excess << ").\n";
                    gpd_cdf_val = 1.0; // Assign default value
                }

            } else {
                // No excesses, so GPD_CDF is zero
                fit = F_star_u;
                // std::cout << "Threshold " << i << ": fit = " << fit << std::endl; // Debugging line

                // std::cout << "Fit (at thresh): " << fit;
            }
        }

        if (fit > 0 && fit < 1) {
            total_sum += nseg * (fit * std::log(fit) + (1 - fit) * std::log(1 - fit));
        }
    }

    // Apply normalization
    double c = -std::log(2 * n - 1);
    return 2.0 * c / k * total_sum;
}


// double GPDEdPeltChangePointDetector::get_segment_cost(const Eigen::MatrixXi& partial_sums, const std::vector<double>& thresholds, const std::vector<double>& p_values, int tau1, int tau2, int k, int n, const double* data, double u) {
//     double total_sum = 0.0;
//     int nseg = tau2 - tau1;  // Total number of points in the segment

//     // Compute F_star_u
//     int count_below_u = 0;
//     int count_equal_u = 0;
//     std::vector<double> excesses;

//     for (int j = tau1; j < tau2; ++j) {
//         if (data[j] < u) {
//             count_below_u++;
//         } else if (data[j] == u) {
//             count_equal_u++;
//         } else {
//             excesses.push_back(data[j] - u);
//         }
//     }

//     double F_star_u = (count_below_u + 0.5 * count_equal_u) / nseg;

//     // Fit GPD if there are excesses
//     std::vector<double> gpd_params;
//     if (!excesses.empty()) {
//         std::vector<double> excesses_sorted = excesses; // Copy and sort for caching
//         std::sort(excesses_sorted.begin(), excesses_sorted.end());

//         // Check if we have already cached the GPD parameters for these excesses
//         auto it = gpd_cache.find(excesses_sorted);
//         if (it == gpd_cache.end()) {
//             // Fit GPD to all the excesses and cache the result
//             fit_gpd(excesses, gpd_params);
//             gpd_cache[excesses_sorted] = gpd_params;
//         } else {
//             gpd_params = it->second;  // Retrieve from cache
//         }
//     }

//     // Now loop over the thresholds using the segment partial sums and p-values
//     for (int i = 0; i < k; ++i) {
//         double t_k = thresholds[i];  // Use global thresholds derived from partial sums
//         double p_val = p_values[i];  // Get the corresponding p-value

//         if (p_val > 0.95) {  // We're only interested in p-values greater than 0.95
//             double fit = 0.0;
//             if (t_k <= u) {
//                 // Use partial sums for the segment
//                 int actual_sum = partial_sums(i, tau2) - partial_sums(i, tau1);
//                 fit = actual_sum * 0.5 / nseg;

//             } else {
//                 // t_k > u
//                 if (!excesses.empty()) {
//                     // **A. Print GPD Parameters Before CDF Calculation**
//                     // std::cout << "GPD Parameters: ";
//                     // for (size_t idx = 0; idx < gpd_params.size(); ++idx) {
//                     //     std::cout << "Param[" << idx << "] = " << gpd_params[idx] << " ";
//                     // }
//                     // std::cout << "\n";

//                     // **B. Print All Excesses Used to Fit GPD**
//                     // std::cout << "Excesses used for GPD fitting (" << excesses.size() << " values): ";
//                     // for (size_t idx = 0; idx < excesses.size(); ++idx) {
//                     //     std::cout << excesses[idx] << " ";
//                     // }
//                     // std::cout << "\n";

//                     // std::cout << "t_k value: " << t_k << "\n";

//                     // **C. Calculate GPD CDF Value**
//                     double excess = *std::min_element(excesses.begin(), excesses.end(), [t_k, u](double a, double b) {
//                         return std::abs(a - (t_k - u)) < std::abs(b - (t_k - u));
//                     });

//                     // **D. Validate Excess Before CDF Calculation**
//                     double gpd_cdf_val;
//                     if (excess < 0) {
//                         // std::cerr << "Error: Negative excess encountered (t_k - u = " << excess << "). Setting gpd_cdf_val to 0.\n";
//                         gpd_cdf_val = 0.0;
//                     } else {
//                         gpd_cdf_val = gpd_cdf(excess, gpd_params);  // GPD CDF at t_k - u
//                     }

//                     // **E. Print gpd_cdf_val After Calculation**
//                     // std::cout << "GPD CDF Value for excess (" << excess << "): " << gpd_cdf_val << "\n";

//                     // **F. Calculate Fit Using F_star_u and gpd_cdf_val**
//                     fit = F_star_u + (1.0 - F_star_u) * gpd_cdf_val;

//                     // **G. Print Fit Values**
//                     // std::cout << "Fit (above threshold): " << fit << "\n";
//                     // std::cout << "Fit (at threshold F_star_u): " << F_star_u << "\n";

//                     // **H. Optional: Check for NaN in gpd_cdf_val**
//                     if (std::isnan(gpd_cdf_val)) {
//                         // std::cerr << "Error: GPD CDF returned NaN for excess (" << excess << ").\n";
//                         // Handle the error, e.g., assign a default value or skip this threshold
//                         gpd_cdf_val = 1.0;
//                     }

//                 } else {
//                     // No excesses, so GPD_CDF is zero
//                     fit = F_star_u;
//                 }
//             }

//             if (fit > 0 && fit < 1) {
//                 total_sum += nseg*(fit * std::log(fit) + (1 - fit) * std::log(1 - fit));
//             }
//         }
//     }

//     // Apply normalization
//     double c = -std::log(2 * n - 1);
//     return 2.0 * c / k * total_sum;
// }


// Function to fit a GPD to excesses
void GPDEdPeltChangePointDetector::fit_gpd(const std::vector<double>& excesses, std::vector<double>& gpd_params) {
    ++gpd_fit_count; // Increment the counter each time GPD fitting is performed

    // Initial parameter estimates
    double shape_init = 0.1;
    double scale_init = std::accumulate(excesses.begin(), excesses.end(), 0.0) / excesses.size();

    // Define the negative log-likelihood function
    auto neg_log_likelihood = [&excesses](const double* params) {
        double shape = params[0];
        double scale = params[1];
        if (scale <= 0) return std::numeric_limits<double>::infinity();

        double log_likelihood = 0.0;
        for (double x : excesses) {
            double term = 1.0 + shape * x / scale;
            if (term <= 0) return std::numeric_limits<double>::infinity();
            log_likelihood += std::log(scale) + (1 + 1 / shape) * std::log(term);
        }
        return log_likelihood;
    };

    // Parameters and settings for optimization
    const int max_iter = 1000;
    const double tol = 1e-5;
    double params[2] = {shape_init, scale_init};
    double best_params[2] = {shape_init, scale_init};
    double min_nll = std::numeric_limits<double>::infinity();

    // Optimization loop (gradient descent placeholder)
    double learning_rate = 0.01; // Set an appropriate learning rate
    for (int iter = 0; iter < max_iter; ++iter) {
        double current_nll = neg_log_likelihood(params);

        if (current_nll < min_nll) {
            min_nll = current_nll;
            best_params[0] = params[0];
            best_params[1] = params[1];
        }

        // Calculate numerical gradient (finite differences)
        double grad_shape, grad_scale;

        // Temporary parameter array for gradient calculation
        double temp_params[2];

        // Gradient with respect to shape
        temp_params[0] = params[0] + tol;
        temp_params[1] = params[1];
        grad_shape = (neg_log_likelihood(temp_params) - current_nll) / tol;

        // Gradient with respect to scale
        temp_params[0] = params[0];
        temp_params[1] = params[1] + tol;
        grad_scale = (neg_log_likelihood(temp_params) - current_nll) / tol;

        // Update parameters using gradient descent
        params[0] -= learning_rate * grad_shape;
        params[1] -= learning_rate * grad_scale;

        // Convergence check
        if (fabs(params[0] - best_params[0]) < tol && fabs(params[1] - best_params[1]) < tol) {
            break;
        }
    }

    // Store the estimated parameters
    gpd_params.push_back(best_params[0]);
    gpd_params.push_back(best_params[1]);
}

// Utility function to compute the GPD CDF
double GPDEdPeltChangePointDetector::gpd_cdf(double x, const std::vector<double>& gpd_params) {
    double shape = gpd_params[0];
    double scale = gpd_params[1];

    if (shape == 0) {
        return 1 - std::exp(-x / scale);
    } else {
        return 1 - std::pow(1 + shape * x / scale, -1.0 / shape);
    }
}

// Function to get the GPD fit count
int GPDEdPeltChangePointDetector::get_gpd_fit_count() const {
    return gpd_fit_count;
}
