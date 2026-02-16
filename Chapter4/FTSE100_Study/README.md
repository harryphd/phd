# Bayesian Clustering of FTSE100 Time Series Extremes (FTSE100 Study)

This repository contains Python code to analyse **FTSE100 constituent time series** by transforming returns into **extreme exceedances** and running an **MCMC sampler** that clusters series within each time window using **GPD (Generalized Pareto) likelihoods**.

The sampler updates:

- changepoints / window boundaries (via updates to `Z`, then recovered as `Taus`)
- cluster allocations `Z`
- mixture weights `alpha`
- GPD parameters `(xi, sigma)` for each active component
- (optionally) split/merge moves to change the number of active components per window (RJ step)


## Running the FTSE100 study


The script loops over multiple runs (`Sim_Final_1` to `Sim_Final_5` by default) and writes output to:

`Data/Sim_Final_<id>/`
It also saves the exceedance matrix as `DataMat.npy`

## What this code does (high level)
Loads FTSE data from FinanceData2.csv, drops the date column and transposes so each row is a company and each column is time.

Builds exceedances using a moving quantile threshold and sets non-exceedances to `NaN`.

Initialises a latent allocation matrix `Z` across time windows and series.

Runs MCMC iterations using `Iterator.Iterate(...)` to update taus, GPD parameters, allocations, and (optionally) the number of components.

## What to change (the main knobs)
### 1) FTSE dataset and preprocessing
In `main.py`, the FTSE data is loaded from:

`FinanceData2.csv`

To use a different file:

change:

`FinData = pd.read_csv("FinanceData2.csv")`
To use a subset of companies or dates:

filter the dataframe before transposing (there is an example of date subsetting in the `__main__` block of `DataGeneration.py`).

### 2) Exceedance thresholding (most important FTSE-specific knob)
The MCMC is run on exceedances, not raw returns.

In `DataGeneration.py`, `extremeFinData(...)` currently constructs exceedances using:

`moving_80(...)`, it uses a 60-day moving 80th percentile threshold with custom edge behaviour. Non-exceedances are set to `NaN`.

If you want to change the thresholding method, you have two good options:

Option A (recommended): use the general threshold builder already implemented
`DataGenerator` includes:

`build_exceedances(data, method={"moving","fixed"}, q, window_size)`
To switch to a 90% moving threshold with a 120-day window, replace:

`FinData_noTime = self.moving_80(FinData_noTime)`
with something like:

`FinData_noTime, u = self.build_exceedances(
    FinData_noTime, method="moving", q=0.9, window_size=120
)`
Option B: edit `moving_80(...)` directly
Inside `moving_80(...)` you can change:

`window_size = 60`

the percentile used (currently 80)

### 3) Number of time windows
In `main.py`:

`NumTaus = 3`
This affects the initial windowing used to create the initial `Z` (`tau_prop` is an even split of the time axis).

### 4) Priors (regularisation / shrinkage)
In `main.py` (inside `Data_Zs()`), you can change:

`pri_nu`, `pri_kappa` for xi

`pri_alpha`, `pri_beta` for sigma

`pri_delt` for the Dirichlet concentration used in alpha updates

Example (current values in the FTSE script):

`pri_nu = 0.1`
`pri_kappa = 2`
`pri_alpha = 3`
`pri_beta = 40`
`pri_delt = 8`
### 5) Proposal step sizes (tuning / mixing)
In `main.py`:

`xi_sig = 0.05`
`scale_sig = 0.05`
These control MH proposals in the GPD update:

`xi_new = xi + Normal(0, xi_sig)`

`sig_new = sig * exp(Normal(0, scale_sig))`

### 6) MCMC run length, burn-in, and how much you save
In `main.py`:

`Num_Iter = 50000`
`burnin = 25000`
The script saves the last `2000 Z` matrices to:

`Data/<Sim>/data_Zs.npy`
To save more/fewer Z draws, change:

`n_save_z = 2000`
### 7) Initialisation of components and weights
In `main.py`:

GPDParams is initialised to the same (xi, sigma) pair for all components (currently `(0.2, 2)`).

`alp_tot` is initialised with two active components per window set to `0.5, 0.5`.

If you want more starting components per window:

initialise additional entries of `alp_tot` to non-zero

and ensure corresponding entries of GPDParams are non-zero too.

### 8) Output naming / number of FTSE runs
In `main.py`, the FTSE script loops:

`for sim_index in range(1, 6):
    Sim_Num = f"Sim_Final_{sim_index}"`
Outputs go under:

`Data/Sim_Final_<sim_index>/`
Change the `range(...)` (or `Sim_Num`) to control how many runs you produce and how they’re named.

Update_Dims.py
Split/merge RJ step to change the number of components in a window.
