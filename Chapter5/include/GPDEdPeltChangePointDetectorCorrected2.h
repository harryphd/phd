#ifndef GPDEDPELTCHANGEPOINTDETECTORCORRECTED2_H
#define GPDEDPELTCHANGEPOINTDETECTORCORRECTED2_H

#include <vector>
#include <map>
#include <Eigen/Dense>

// Structure to hold the result of change point detection
struct ChangePointResultGPD {
    int* change_points;  // Array of change point indexes
    double total_cost;   // Total cost associated with the change points
};

// Structure to hold partial sums, thresholds, and p-values
struct PartialSumsResult {
    Eigen::MatrixXi partial_sums;         // Matrix of partial sums
    std::vector<double> thresholds;       // Vector of thresholds
    std::vector<double> p_values;         // Vector of corresponding p-values (added for p-values above 0.95)
};

// Class definition for GPDEdPeltChangePointDetector
class GPDEdPeltChangePointDetector {
public:
    // Constructor
    GPDEdPeltChangePointDetector();

    // Destructor
    ~GPDEdPeltChangePointDetector();

    // Main function to get change point indexes and total cost
    ChangePointResultGPD get_change_point_indexes(const double* data, int n, double penalty_n, int min_distance, int& out_cpts_count);

    // Function to get the GPD fit count
    int get_gpd_fit_count() const;

    // **Added Public Functions**
    /**
     * @brief Function to calculate partial sums, thresholds, and p-values.
     * 
     * @param data Pointer to the array of data points.
     * @param n    Number of data points.
     * @param k    Number of thresholds.
     * @return PartialSumsResult The computed partial sums, thresholds, and p-values.
     */
    PartialSumsResult get_partial_sums(const double* data, int n, int k);

    /**
     * @brief Calculates the cost of a segment based on partial sums and thresholds.
     * 
     * @param partial_sums The matrix of partial sums.
     * @param thresholds    Vector of threshold values.
     * @param p_values      Vector of p-values corresponding to the thresholds.
     * @param tau1         Start index of the segment (inclusive).
     * @param tau2         End index of the segment (exclusive).
     * @param k            Number of thresholds.
     * @param n            Total number of data points.
     * @param data         Pointer to the array of data points.
     * @param u            95th percentile value.
     * @return double       The calculated cost of the segment.
     */
    double get_segment_cost(const Eigen::MatrixXi& partial_sums, const std::vector<double>& thresholds, const std::vector<double>& p_values, int tau1, int tau2, int k, int n, const double* data, double u);

private:
    // Member variables
    int gpd_fit_count;

    // Thread-local GPD cache
    static thread_local std::map<std::vector<double>, std::vector<double>> gpd_cache;

    // Utility function to find the index of the minimum element in an array
    int which_min(const double* values, int size);

    // Function to fit a GPD to excesses
    void fit_gpd(const std::vector<double>& excesses, std::vector<double>& gpd_params);

    // Utility function to compute the GPD CDF
    double gpd_cdf(double x, const std::vector<double>& gpd_params);
};

#endif // GPDEDPELTCHANGEPOINTDETECTORCORRECTED2_H
