#!/bin/bash

python src/main.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 400 --split_variance 0 --M 2 --recursion 1 --optim_type Boyd --lam 5e-2 --results_fldr_name center_storm --data_path ../data/out_two --data_fname center_storm_0_final_data.csv --results_filename center_storm_0.csv 