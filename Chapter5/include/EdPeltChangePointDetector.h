#ifndef EDPELTCHANGEPOINTDETECTOR_H
#define EDPELTCHANGEPOINTDETECTOR_H

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <Eigen/Dense>

using namespace Eigen;

// Struct to encapsulate the change points and the total cost for ED-PELT
struct ChangePointResult {
    int* change_points;
    double total_cost;
};

class EdPeltChangePointDetector {
public:
    EdPeltChangePointDetector() = default;
    ~EdPeltChangePointDetector() = default;

    static ChangePointResult get_change_point_indexes(const double* data, int n, double penalty_n, int min_distance, int& out_cpts_count);
    static MatrixXi get_partial_sums(const double* data, int n, int k);
    static double get_segment_cost(const MatrixXi& partial_sums, int tau1, int tau2, int k, int n);

private:
    static int which_min(const double* values, int size);
};

#endif // EDPELTCHANGEPOINTDETECTOR_H
