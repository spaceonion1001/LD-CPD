#!/bin/bash

python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_0_unprocessed_data.csv \
    --results_filename center_storm_0.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_1_unprocessed_data.csv \
    --results_filename center_storm_1.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_2_unprocessed_data.csv \
    --results_filename center_storm_2.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_3_unprocessed_data.csv \
    --results_filename center_storm_3.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_4_unprocessed_data.csv \
    --results_filename center_storm_4.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_5_unprocessed_data.csv \
    --results_filename center_storm_5.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_6_unprocessed_data.csv \
    --results_filename center_storm_6.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_7_unprocessed_data.csv \
    --results_filename center_storm_7.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_8_unprocessed_data.csv \
    --results_filename center_storm_8.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_9_unprocessed_data.csv \
    --results_filename center_storm_9.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_10_unprocessed_data.csv \
    --results_filename center_storm_10.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_11_unprocessed_data.csv \
    --results_filename center_storm_11.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_12_unprocessed_data.csv \
    --results_filename center_storm_12.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_13_unprocessed_data.csv \
    --results_filename center_storm_13.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_14_unprocessed_data.csv \
    --results_filename center_storm_14.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_15_unprocessed_data.csv \
    --results_filename center_storm_15.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_16_unprocessed_data.csv \
    --results_filename center_storm_16.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_17_unprocessed_data.csv \
    --results_filename center_storm_17.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_18_unprocessed_data.csv \
    --results_filename center_storm_18.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_19_unprocessed_data.csv \
    --results_filename center_storm_19.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_20_unprocessed_data.csv \
    --results_filename center_storm_20.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_21_unprocessed_data.csv \
    --results_filename center_storm_21.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_22_unprocessed_data.csv \
    --results_filename center_storm_22.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_23_unprocessed_data.csv \
    --results_filename center_storm_23.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_24_unprocessed_data.csv \
    --results_filename center_storm_24.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_25_unprocessed_data.csv \
    --results_filename center_storm_25.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_26_unprocessed_data.csv \
    --results_filename center_storm_26.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_27_unprocessed_data.csv \
    --results_filename center_storm_27.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_28_unprocessed_data.csv \
    --results_filename center_storm_28.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_29_unprocessed_data.csv \
    --results_filename center_storm_29.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_30_unprocessed_data.csv \
    --results_filename center_storm_30.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_31_unprocessed_data.csv \
    --results_filename center_storm_31.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_32_unprocessed_data.csv \
    --results_filename center_storm_32.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_33_unprocessed_data.csv \
    --results_filename center_storm_33.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_34_unprocessed_data.csv \
    --results_filename center_storm_34.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_35_unprocessed_data.csv \
    --results_filename center_storm_35.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_36_unprocessed_data.csv \
    --results_filename center_storm_36.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_37_unprocessed_data.csv \
    --results_filename center_storm_37.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 0 --data mesonet --single_test 1 --step_size 1 --window_size 600 --M 2  \
     --lam 5e-2 --results_fldr_name center_storm_kesh_clime --data_path ../data/out_center_raw --data_fname center_storm_storm_38_unprocessed_data.csv \
    --results_filename center_storm_38.csv --train_percent 0.1 --burn_in 600 --post_window_size 25 --alt 1 --estimator clime --auto_lambda 1