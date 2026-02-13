# Bayesian Clustering of Time Series Extremes - Simulation Study

This repository contains Python code to **simulate multiple time series with changepoints** and **Generalized Pareto (GPD) emissions**, then run an **MCMC sampler** that updates:

- changepoints `Taus`
- cluster allocations `Z`
- mixture weights `alpha`
- GPD parameters `(xi, sigma)`
- (optionally) split/merge moves to change the number of mixture components per time window (RJ step)

**Entry point:** `main.py`

---

## Quick start

```bash
python main.py
```
## What to change (the main knobs)

This section is the “how do I adjust it?” guide — the variables below are the ones you’ll most likely edit.

### 1) Simulation size: number of series and time length

In main.py (inside Data_Zs() or the Data_Zs_mod(...) override in main()), change:

K = number of time series

T = length of each time series

Example:

K = 10
T = 1000


If you change K, you must also update the lengths of InputShapes[...] and InputScales[...] (see below).

### 2) Changepoints / number of time windows

Still in main.py, set:

Taus = list of changepoints (commonly ends with T)

NumTaus = number of time windows

Example:

Taus = [280, 600, T]
NumTaus = 3


Rule of thumb: if you change NumTaus, also update:

Taus

InputShapes

InputScales

### 3) “True” GPD parameters used to generate the synthetic data

In main.py, the synthetic generator is given per-window “means” for:

xi (“shape”) via InputShapes

sigma (“scale”) via InputScales

They are lists of length NumTaus, and each entry contains a list of length K.

Example structure:

InputShapes = [
  [...],  # window 1: length K
  [...],  # window 2: length K
  [...],  # window 3: length K
]
InputScales = [
  [...],  # window 1: length K
  [...],  # window 2: length K
  [...],  # window 3: length K
]


If you change K, you must update every inner list in InputShapes and InputScales to length K.

### 4) How the synthetic cluster labels Z are generated

In DataGeneration.py, Z labels are currently assigned per time window with:

random.choice([k*K, k*K+1])


So each window is picking between two possible component labels (k*K and k*K+1).

If you want more than 2 components per window, expand that list, e.g.

random.choice([k*K, k*K+1, k*K+2])


Also note: the generator currently defines window boundaries using an even split:

tau_prop = [int((T/NumTaus)*i) for i in range(NumTaus)] + [T]


If you want the generation windows to align exactly with Taus, you’d modify this part to use Taus directly.

### 5) Prior settings (regularization / shrinkage)

In main.py you can change the priors:

pri_nu, pri_kappa: prior for xi

pri_alpha, pri_beta: prior for sigma

pri_delt: Dirichlet concentration added to counts for mixture weights alpha

Example:

pri_nu = 0.1
pri_kappa = 2
pri_alpha = 3
pri_beta = 1
pri_delt = 10


### 6) Proposal step sizes (tuning / mixing)

In main.py:

xi_sig: proposal SD for xi updates

scale_sig: log-scale proposal SD for sigma updates

Example:

xi_sig = 0.1
scale_sig = 0.25

### 7) MCMC run length, burn-in, and saving samples

In run_simulation() in main.py:

num_iter: total iterations

burnin: burn-in cutoff

save_after_iter: when to start saving Z samples

num_z_to_save: how many Z samples to save

data_zs_path: output filename for saved Z samples

Example:

num_iter = 80000
burnin = 40000

save_after_iter = 3000
num_z_to_save = 2000
data_zs_path = "data_Zs.npy"

### 8) Output naming

In main.py the simulation folder name is typically set by something like:

sim_num = f"Sim_{i + 80}"


Outputs go under:

Data/Sim_<...>/
