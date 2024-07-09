#!/bin/bash

python src/main.py --sim 1 --sim_type orthogonal_block_fix_window --single_test 0 --step_size 1 --window_size 50 --split_variance 0 --M 4 --dim 20 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
    --candidate_recursion 0 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 50 --train_percent 0.1 --thav 0 --thresh_const 0.8 --fix_pre 0
python src/main.py --sim 1 --sim_type orthogonal_block_fix_window --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 4 --dim 40 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
    --candidate_recursion 0 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 100 --train_percent 0.1 --thav 0 --thresh_const 0.8 --fix_pre 0
python src/main.py --sim 1 --sim_type orthogonal_block_fix_window --single_test 0 --step_size 1 --window_size 150 --split_variance 0 --M 4 --dim 60 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
    --candidate_recursion 0 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 150 --train_percent 0.1 --thav 0 --thresh_const 0.8 --fix_pre 0
python src/main.py --sim 1 --sim_type orthogonal_block_fix_window  --single_test 0 --step_size 1 --window_size 200 --split_variance 0 --M 4 --dim 80 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
    --candidate_recursion 0 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 200 --train_percent 0.1 --thav 0 --thresh_const 0.8 --fix_pre 0

python src/main.py --sim 1 --sim_type cai_model_one_fix_window --single_test 0 --step_size 1 --window_size 50 --split_variance 0 --M 4 --dim 20 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
    --candidate_recursion 0 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 50 --train_percent 0.1 --thav 0 --thresh_const 0.9 --fix_pre 0
python src/main.py --sim 1 --sim_type cai_model_one_fix_window --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 4 --dim 40 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
    --candidate_recursion 0 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 100 --train_percent 0.1 --thav 0 --thresh_const 0.9 --fix_pre 0
python src/main.py --sim 1 --sim_type cai_model_one_fix_window --single_test 0 --step_size 1 --window_size 150 --split_variance 0 --M 4 --dim 60 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
    --candidate_recursion 0 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 150 --train_percent 0.1 --thav 0 --thresh_const 0.9 --fix_pre 0
python src/main.py --sim 1 --sim_type cai_model_one_fix_window --single_test 0 --step_size 1 --window_size 200 --split_variance 0 --M 4 --dim 80 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
    --candidate_recursion 0 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 200 --train_percent 0.1 --thav 0 --thresh_const 0.9 --fix_pre 0

# # AVESENOV TESTING 

# python src/avesenov_cpd.py --sim 1 --sim_type orthogonal_small --window_size 50 --M 4 --dim 20 --N 500 --sim_scale 0.8 --train_percent 0.1 --single_test 0
# python src/avesenov_cpd.py --sim 1 --sim_type orthogonal_small --window_size 100 --M 4 --dim 40 --N 500 --sim_scale 0.8 --train_percent 0.1 --single_test 0
# python src/avesenov_cpd.py --sim 1 --sim_type orthogonal_small --window_size 150 --M 4 --dim 60 --N 500 --sim_scale 0.8 --train_percent 0.1 --single_test 0
# python src/avesenov_cpd.py --sim 1 --sim_type orthogonal_small --window_size 200 --M 4 --dim 80 --N 500 --sim_scale 0.8 --train_percent 0.1 --single_test 0

# # AVESENOV TESTING

# python src/avesenov_cpd.py --sim 1 --sim_type cai_model_one --window_size 50 --M 4 --dim 20 --N 500 --sim_scale 0.8 --train_percent 0.1 --single_test 0
# python src/avesenov_cpd.py --sim 1 --sim_type cai_model_one --window_size 100 --M 4 --dim 40 --N 500 --sim_scale 0.8 --train_percent 0.1 --single_test 0
# python src/avesenov_cpd.py --sim 1 --sim_type cai_model_one --window_size 150 --M 4 --dim 60 --N 500 --sim_scale 0.8 --train_percent 0.1 --single_test 0
# python src/avesenov_cpd.py --sim 1 --sim_type cai_model_one --window_size 200 --M 4 --dim 80 --N 500 --sim_scale 0.8 --train_percent 0.1 --single_test 0

