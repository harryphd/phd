#include "EdPeltChangePointDetector.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <stdexcept>

// Utility function to find the index of the minimum element in an array
int EdPeltChangePointDetector::which_min(const double* values, int size) {
    int min_index = 0;
    for (int i = 1; i < size; ++i) {
        if (values[i] < values[min_index]) {
            min_index = i;
        }
    }
    return min_index;
}

// Main function to get change point indexes and total cost
ChangePointResult EdPeltChangePointDetector::get_change_point_indexes(const double* data, int n, double penalty_n, int min_distance, int& out_cpts_count) {
    if (n <= 2 || min_distance < 1 || min_distance > n) {
        throw std::invalid_argument("Invalid input size or minimum distance.");
    }

    double penalty = penalty_n; //* std::log(static_cast<double>(n));
    int k = std::min(n, static_cast<int>(std::ceil(4 * std::log(static_cast<double>(n)))));
    MatrixXi partial_sums = get_partial_sums(data, n, k);

    double* best_cost = new double[n + 1]();
    int* previous_change_point_index = new int[n + 1]();
    int* previous_taus = new int[n + 1]();
    int previous_taus_size = 0;

    best_cost[0] = -penalty;
    previous_taus[previous_taus_size++] = 0;

    for (int current_tau = min_distance; current_tau < 2 * min_distance; ++current_tau) {
        best_cost[current_tau] = get_segment_cost(partial_sums, 0, current_tau, k, n);
    }

    previous_taus[previous_taus_size++] = min_distance;
    double* cost_for_previous_tau = new double[n];

    for (int current_tau = 2 * min_distance; current_tau <= n; ++current_tau) {
        for (int i = 0; i < previous_taus_size; ++i) {
            int previous_tau = previous_taus[i];
            cost_for_previous_tau[i] = best_cost[previous_tau] +
                get_segment_cost(partial_sums, previous_tau, current_tau, k, n) + penalty;
        }

        int best_previous_tau_index = which_min(cost_for_previous_tau, previous_taus_size);
        best_cost[current_tau] = cost_for_previous_tau[best_previous_tau_index];
        previous_change_point_index[current_tau] = previous_taus[best_previous_tau_index];

        double current_best_cost = best_cost[current_tau];
        int new_size = 0;
        for (int i = 0; i < previous_taus_size; ++i) {
            int tau = previous_taus[i];
            if (best_cost[tau] + get_segment_cost(partial_sums, tau, current_tau, k, n) + penalty < current_best_cost + penalty) {
                previous_taus[new_size++] = tau;
            }
        }
        previous_taus[new_size++] = current_tau - (min_distance - 1);
        previous_taus_size = new_size;
    }

    int* change_point_indexes = new int[n];
    int change_point_count = 0;
    for (int current_index = previous_change_point_index[n]; current_index != 0; current_index = previous_change_point_index[current_index]) {
        change_point_indexes[change_point_count++] = current_index - 1;
    }
    std::reverse(change_point_indexes, change_point_indexes + change_point_count);
    out_cpts_count = change_point_count;

    // Calculate the total cost of all segments with the final changepoints
    double total_cost = 0.0;
    int previous_tau = 0;
    for (int i = 0; i < change_point_count; ++i) {
        int current_tau = change_point_indexes[i] + 1;
        total_cost += get_segment_cost(partial_sums, previous_tau, current_tau, k, n);
        previous_tau = current_tau;
    }
    // Add the cost of the last segment
    total_cost += get_segment_cost(partial_sums, previous_tau, n, k, n);

    // Prepare the result to return
    ChangePointResult result;
    result.change_points = change_point_indexes;
    result.total_cost = total_cost;

    // Cleanup
    delete[] best_cost;
    delete[] previous_change_point_index;
    delete[] previous_taus;
    delete[] cost_for_previous_tau;

    return result;
}

// Function to calculate partial sums
MatrixXi EdPeltChangePointDetector::get_partial_sums(const double* data, int n, int k) {
    MatrixXi partial_sums = MatrixXi::Zero(k, n + 1);

    std::vector<double> sorted_data(data, data + n);
    std::sort(sorted_data.begin(), sorted_data.end());

    for (int i = 0; i < k; ++i) {
        double z = -1 + (2 * i + 1.0) / k;
        double p = 1.0 / (1 + std::pow(2 * n - 1, -z));
        double t = sorted_data[static_cast<int>((n - 1) * p)];

        for (int tau = 1; tau <= n; ++tau) {
            partial_sums(i, tau) = partial_sums(i, tau - 1) + (data[tau - 1] < t ? 2 : (data[tau - 1] == t ? 1 : 0));
        }
    }

    return partial_sums;
}

// Function to calculate the cost of a segment
double EdPeltChangePointDetector::get_segment_cost(const MatrixXi& partial_sums, int tau1, int tau2, int k, int n) {
    double total_sum = 0.0;
    int nseg = tau2 - tau1;

    for (int i = 0; i < k; ++i) {
        int actual_sum = partial_sums(i, tau2) - partial_sums(i, tau1);
        if (actual_sum != 0 && actual_sum != nseg * 2) {
            double fit = actual_sum * 0.5 / nseg;
            if (fit > 0 && fit < 1) {
                total_sum += nseg * (fit * std::log(fit) + (1 - fit) * std::log(1 - fit));
            }
        }
    }

    double c = -std::log(2 * n - 1);
    return 2.0 * c / k * total_sum;
}
