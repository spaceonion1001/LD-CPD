# Online Precision Matrix Changepoint Detection with Localization to Groups of Dimensions

## Overview

This repository contains the code for the paper "Online Precision Matrix Changepoint Detection with Localization to Groups of Dimensions", currently in review at DMKD.

### Requirements

The requirements are outlines in the `requirements.txt` file. A working R installation is also required for the XCC and KM/KMA algorithms. 

## Examples

### Simulations

To run simulations, follow the example provided, and change the `--sim_type` argument appropriately.

```bash
python src/main.py --sim 1 --sim_type orthogonal_small --single_test 0 --step_size 1 --window_size 50 --M 4 --dim 20 --optim_type Boyd --N 500 --recursion 1 --linkage single \
    --candidate_recursion 0 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 20 --thav 0 --thresh_const 0.8
```

### Real-World Data

To run with real-world data, follow the example provided using the `mesonet` data, and change the `--data` argument. To handle new datasets, another case should be added to the `resolve_data()` function in `main.py` and loaded from `utils.py`.

```bash
python src/main.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --split_variance 0 --M 2 --recursion 1 --candidate_recursion 1 \
    --optim_type Boyd --results_fldr_name center_storm_unprocessed --data_path ../data/out_center_raw --data_fname center_storm_storm_1_unprocessed_data.csv \
    --results_filename center_storm_1.csv --save_test_stat 1 --base_M 4 --log_pvals 1 --post_window_size 25 --linkage single --thresh_const 1.2
```

## Replicating Paper Results

To replicate the calculations for truncated averaged AMOC AUC values, along with the plots and statistical tests, please follow the sections below. One section is for the simulation studies, and the following for the real-world case studies. The implementations of XCC [1] and KM/KMA [2,3] are found in their own files `xia_cpd_working.py` and `kesh_cpd.py` respectively. These implementations utilize R packages for precision matrix estimations in our best attempt to follow the algorithmic procedures listed in the papers. All collection scripts should be ran before collating the results.

### Simulations

To collect the simulation results presented in the paper, along with the Appendix, first run the following, for each algorithm. These cover six different simulation models. Results are stored in `results/simulation_results/`.

#### LD-CPD

```bash
bash ldcpd_sims.sh
```

#### KM/KMA

```bash
bash kesh_sims.sh
```

#### XCC

```bash
bash cai_sims.sh
```

### Mesonet

To collect the results from OK Mesonet data [4,5], run the following scripts. Each section processes the different experiments as outlined in the paper (Center and Pressure). Results are stored in `results/center_storm_unprocessed/` and `results/pressure_storm_unprocessed/` respectively. This data is not public and would need to be purchased, but the details are covered in the paper. The storm collection process for "ground-truth" values, is also covered.

#### Center

##### LD-CPD

```bash
bash mesonet.sh
```

##### KM/KMA

```bash
bash mesonet_kesh_clime.sh
bash mesonet_kesh_alt_clime.sh
```

##### XCC

```bash
bash mesonet_cai.sh
```

#### Pressure

##### LD-CPD

```bash
bash mesonet_pressure.sh
```

##### KM/KMA

```bash
bash mesonet_pressure_kesh_clime.sh
bash mesonet_pressure_kesh_alt_clime.sh
```

##### XCC

```bash
bash mesonet_pressure_cai.sh
```

### Results Calculations

To calculate the values for the tables, and the figures from Simulations and Mesonet experiments, run the following commands. Average AMOC figures are saved to `amoc_figs/`. The AUC confidence interval and boxplot figures are saved to `lrt_test_figs/`. Wilcoxon test results for simulations are saved to `results/simulation_results/`, and for Mesonet to `amoc_figs/mesonet/`.

```bash
python src/amoc_gen.py --sims --mesonet --clime
```

```bash
python src/wilcoxon_test.py --clime
```

### S&P 500

To calculate results on the S&P 500 data, please run the following script. Results are stored in `results/sap_results/`. This data is publicly available, and the details and process are found in the paper. We have included the `sap_scaled_returns.csv` in this directory.

```bash
bash sap.sh
```

Plots can be produced by running the plotting script.

```bash
python src/sap_plots.py --clime
```

## References

[1] Y. Xia, T. Cai, and T. T. Cai, "Testing differential networks with applications to detecting gene-by-gene interactions," *Biometrika*, vol. 102, pp. 247–266, 2015.

[2] H. Keshavarz, G. Michailidis, and Y. Atchadé, "Sequential change-point detection in high-dimensional Gaussian graphical models," *Journal of Machine Learning Research*, vol. 21, pp. 1–57, 2020.

[3] H. Keshavarz and G. Michailidis, "Online detection of local abrupt changes in high-dimensional Gaussian graphical models," *arXiv preprint arXiv:2003.06961*, 2020.

[4] R. A. McPherson, C. Fiebrich, K. C. Crawford, R. L. Elliott, J. R. Kilby, D. L. Grimsley, J. E. Martinez, J. B. Basara, B. G. Illston, D. A. Morris, K. A. Kloesel, S. J. Stadler, A. D. Melvin, A. J. Sutherland, and H. Shrivastava, "Statewide monitoring of the mesoscale environment: A technical update on the Oklahoma Mesonet," *J. Atmos. Oceanic Technol.*, vol. 24, pp. 301–321, 2007.

[5] F. V. Brock, K. C. Crawford, R. L. Elliott, G. W. Cuperus, S. J. Stadler, H. L. Johnson, and M. D. Eilts, "The Oklahoma Mesonet: A technical overview," *J. Atmos. Oceanic Technol.*, vol. 12, pp. 5–19, 1995.
