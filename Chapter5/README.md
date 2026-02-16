# ED-PELT and GPD-ED-PELT (C++) — changepoint detection + CROPs comparison

This folder contains a small C++ implementation of:

- **ED-PELT**: PELT changepoint detection using an *empirical-distribution* cost (via partial sums).
- **GPD-ED-PELT**: a modified ED-PELT cost that, **above a high threshold `u`**, models the tail using a **Generalized Pareto Distribution (GPD)** fitted to segment excesses.
- A **CROPs**-style penalty sweep (searching across penalty values) to compare ED-PELT vs GPD-ED-PELT on simulated data.

The layout is:
- `.cpp` files in `src/`
- header files in `include/`

include/
EdPeltChangePointDetector.h
GPDEdPeltChangePointDetectorCorrected2.h
src/
EdPeltChangePointDetector.cpp
GPDEdPeltChangePointDetectorCorrected.cpp
main_PELT_comparison_w_CROPs.cpp


---

## Dependencies

### Eigen (required)
Both detectors use **Eigen** (`#include <Eigen/Dense>`). Install `Eigen` and ensure your compiler can find it.

## Build
From the project root:

```
mkdir -p build
g++ -O3 -std=c++17 -Iinclude -I/usr/include/eigen3 \
  src/EdPeltChangePointDetector.cpp \
  src/GPDEdPeltChangePointDetectorCorrected.cpp \
  src/main_PELT_comparison_w_CROPs.cpp \
  -o build/pelt_compare -pthread
```

Then run:
```
./build/pelt_compare
```
It will write:

`final_results_95.csv`

## What each file does
### ED-PELT
`include/EdPeltChangePointDetector.h` + `src/EdPeltChangePointDetector.cpp`
Implements ED-PELT.

Main API:
```
static ChangePointResult
get_change_point_indexes(const double* data,
                         int n,
                         double penalty_n,
                         int min_distance,
                         int& out_cpts_count);
```
`data`: pointer to time series values

`n`: number of observations

`penalty_n`: penalty term (often proportional to `log(n)`)

`min_distance`: minimum spacing between changepoints (enforced by the recursion)

`out_cpts_count`: (output) number of changepoints returned

Returns a ChangePointResult:

`change_points`: dynamically allocated `int*` of changepoint indexes

`total_cost`: total (unpenalized) cost of the segmentation

Memory note: change_points is allocated with `new[]` and must be freed by the caller with `delete[]`.

### GPD-ED-PELT (GPEP)
`include/GPDEdPeltChangePointDetectorCorrected2.h` + `src/GPDEdPeltChangePointDetectorCorrected.cpp`
Implements GPD-ED-PELT: ED partial-sum cost up to a high threshold`u`, and a GPD tail model beyond `u`.

Main API:
```
ChangePointResultGPD
get_change_point_indexes(const double* data,
                         int n,
                         double penalty_n,
                         int min_distance,
                         int& out_cpts_count);
```
Key details (as implemented):

Internally chooses:

`k = min(n, ceil(4 * log(n)))` thresholds for the ED-style cost

`u` as an empirical high quantile (the code uses `~0.959` quantile)

For a segment `[tau1, tau2)`:

Computes `F*(u)` from counts below/at `u`

Fits a GPD to segment excesses `data[j] - u` where `data[j] > u`

Uses the fitted GPD CDF to extend the fitted CDF for thresholds above `u`

Caching:

Uses a thread-local cache keyed by sorted segment excesses to avoid refitting the GPD repeatedly inside the dynamic program.

Memory note: like ED-PELT, change_points is allocated with new[] and must be freed by the caller with delete[].

### Comparison Simulation Study
`src/main_PELT_comparison_w_CROPs.cpp`
Runs a simulation study comparing ED-PELT vs GPD-ED-PELT, using a CROPs-like penalty search.

What it does:

Simulates `num_experiments` time series (default `500`) of length `num_points` (default `5000`)

Each series has a true changepoint (default `2500`)

Runs CROPs over a penalty range to pick a penalty (it uses the largest beta returned)

Runs ED-PELT and GPD-ED-PELT using that selected penalty

Saves a CSV:

each row: "ed changepoints", "gpd-ed changepoints"

Output file:

`final_results_95.csv`

Threading:

Uses `std::thread::hardware_concurrency()` threads.

Protects shared result vectors with a mutex.

## What to change (main knobs)
These are the practical “edit-this-first” settings for the experiments.

1) Simulation size and truth (in `main_PELT_comparison_w_CROPs.cpp`)
Near the bottom:

`int num_points = 5000;`
`int true_changepoint = 2500;`
`int num_experiments = 500;`
Change these to control:

series length

true changepoint location

number of replicated experiments

2) Minimum segment length / changepoint spacing
Both ED-PELT and GPD-ED-PELT are currently called with:

`min_distance = 300`
Search for:

`get_change_point_indexes(..., 300, ...)`
and change 300 to enforce shorter/longer minimum segment lengths.

3) Penalty ranges for CROPs
In `run_experiment_range(...):`

ED-PELT penalty range:

`double beta_min_ed  = 0.2  * log(num_points);`
`double beta_max_ed  = 50.0 * log(num_points);`
GPD-ED-PELT penalty range:

`double beta_min_gpep = 1.0   * log(num_points);`
`double beta_max_gpep = 500.0 * log(num_points);`
If you see under/over-segmentation, widen/narrow these ranges.

4) How the “best” penalty is chosen
Currently:

`double best_penalty = *crops_results.rbegin();`
i.e. the largest penalty in the CROPs result set.

If you want a different selection rule (e.g. median penalty, or pick the penalty giving changepoint count closest to a target), this is the line to change.

5) Tail threshold quantile for GPD-ED-PELT
In `GPDEdPeltChangePointDetectorCorrected.cpp`, `u` is set via:

`double u = sorted_data[(n - 1) * 0.959];`
Change `0.959` to a different high quantile if desired (e.g. `0.95, 0.975, 0.99`).

6) GPD fitting behaviour
`fit_gpd(...)` uses a simple gradient-descent style numerical optimisation over (shape, scale) with:

`const int max_iter = 1000;`
`double learning_rate = 0.01;`
`const double tol = 1e-5;`
If you get unstable fits, this is where you tune:

`learning_rate`

`max_iter`

`stopping tolerance`

(And/or replace the optimiser with a more robust method.)

7) Output filename
At the end of `main():`

`save_to_csv("final_results_95.csv", ...);`
Rename the file here.

Using the detectors directly (minimal snippet)
ED-PELT:

`#include "EdPeltChangePointDetector.h"`
```
int out_count = 0;
auto res = EdPeltChangePointDetector::get_change_point_indexes(x, n, penalty, min_dist, out_count);
// res.change_points is int* length out_count
delete[] res.change_points;
```
GPD-ED-PELT:

`#include "GPDEdPeltChangePointDetectorCorrected2.h"`
```
GPDEdPeltChangePointDetector det;
int out_count = 0;
auto res = det.get_change_point_indexes(x, n, penalty, min_dist, out_count);
delete[] res.change_points;
```
## Notes / gotchas
Remember to `delete[]` the returned change_points array.

`Eigen` must be installed and included via your compiler include path.

CROPs here is implemented as a penalty-interval refinement; it caches `(Qm, m)` for evaluated penalties to reduce repeated work.
