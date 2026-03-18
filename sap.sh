#bin/bash

python src/kesh_cpd.py --sim 0 --sap 1 --data sap --single_test 0 --step_size 1 --window_size 200 --burn_in 200 --results_fldr_name sap_500 --data_path ../data \
    --dim 80 --train_percent 0.1 --post_window_size 22 --alt 0 --estimator clime --auto_lambda 1

python src/kesh_cpd.py --sim 0 --sap 1 --data sap --single_test 0 --step_size 1 --window_size 200 --burn_in 200 --results_fldr_name sap_500 --data_path ../data \
    --dim 80 --post_window_size 22 --alt 1 --estimator clime --auto_lambda 1

python src/main.py --sim 0 --sap 1 --data sap --single_test 0 --step_size 1 --window_size 200 --base_M 2 --recursion 1 --candidate_recursion 1 \
    --optim_type Boyd --results_fldr_name sap_500 --data_path ../data --dim 80 \
    --save_test_stat 1 --base_M 4 --log_pvals 1 --post_window_size 22 --linkage complete --thresh_const 0.5 --fix_pre 1

python src/xia_cpd_working.py --sim 0 --sap 1 --data sap --single_test 0 --step_size 1 --window_size 200 --results_fldr_name sap_500 --data_path ../data \
    --dim 80 --post_window_size 22 --fix_pre 1