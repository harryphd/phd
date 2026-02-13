# EVA Data Challenge — Challenge 1 (Extreme Value Analysis)

This repository contains an R Markdown workflow used for **EVA (Extreme Value Analysis) Data Challenge — Question/Challenge 1**. The approach clusters covariates (by season and meteorological features), fits **GPD tail models** per cluster, and produces **extreme quantile estimates** for the test set using a **mixture-over-clusters** formulation (as described in the Challenge 1 paper).

## Contents

- **`UniofbathtopiaC1Code.Rmd`**  
  Main analysis notebook (R Markdown). Reads the training + test CSVs, performs clustering, fits GPD models, and outputs final extreme quantiles.

- **`Elsom_Pawley_Paper.pdf`**  
  Reference paper describing the Challenge 1 method (mixture over clusters / posterior-weighted tail probability).

## Data

The notebook expects the following files in the **project root**:

- `Amaurot.csv` (training set)
- `AmaurotTestSet.csv` (test set)

If these files are not in the root directory, update the `read.csv(...)` paths inside the Rmd.

## Method Summary

1. **Feature engineering**
   - Converts wind direction/speed into velocity components.
   - Splits observations into **seasons** (S1, S2).

2. **Clustering**
   - Fits a Gaussian mixture model using `mclust` to obtain **posterior cluster probabilities** for each observation.

3. **Tail modelling**
   - Fits **Generalized Pareto Distributions (GPD)** above cluster-specific thresholds using `ismev::gpd.fit`.

4. **Extreme quantiles for test set**
   - Computes the extreme quantile by solving for \( q \) such that the **posterior-weighted mixture tail probability** matches the target level:
     \[
     \sum_{j=1}^K w_j \, P(Y > q \mid Z=j) = \alpha
     \]
     where \(w_j = P(Z=j \mid X=x)\) comes from the clustering model.

5. **Uncertainty (optional in the notebook)**
   - Uses an asymptotic normal approximation to simulate GPD parameters and form central quantile intervals.

## Requirements

R packages used in the notebook include:

- `mclust`
- `ismev`
- `evd`
- `ggplot2`, `ggpubr`
- `corrplot`, `reshape2`, `dplyr`
- `mvtnorm` (for uncertainty simulation)
- `knitr`, `rmarkdown` (for rendering)

Install missing packages in R:
```r
install.packages(c(
  "mclust","ismev","evd","ggplot2","ggpubr",
  "corrplot","reshape2","dplyr","mvtnorm",
  "knitr","rmarkdown"
))

