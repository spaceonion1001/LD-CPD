#!/bin/bash

# python src/main.py --sim 1 --sim_type orthogonal_small --single_test 0 --step_size 1 --window_size 50 --split_variance 0 --M 4 --dim 20 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
#     --candidate_recursion 0 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 20 --train_percent 0.1 --thav 0 --thresh_const 0.8 --save_lambda 1
# python src/main.py --sim 1 --sim_type orthogonal_small --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 4 --dim 40 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
#     --candidate_recursion 0 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 20 --train_percent 0.1 --thav 0 --thresh_const 0.8 --save_lambda 1
# python src/main.py --sim 1 --sim_type orthogonal_small --single_test 0 --step_size 1 --window_size 150 --split_variance 0 --M 4 --dim 60 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
#     --candidate_recursion 0 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 20 --train_percent 0.1 --thav 0 --thresh_const 0.8 --save_lambda 1
# python src/main.py --sim 1 --sim_type orthogonal_small --single_test 0 --step_size 1 --window_size 200 --split_variance 0 --M 4 --dim 80 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
#     --candidate_recursion 0 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 20 --train_percent 0.1 --thav 0 --thresh_const 0.8 --save_lambda 1

# python src/main.py --sim 1 --sim_type orthogonal_hard --single_test 0 --step_size 1 --window_size 50 --split_variance 0 --M 4 --dim 20 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
#     --candidate_recursion 1 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 20 --train_percent 0.1 --thav 0 --thresh_const 0.8 --save_lambda 1
# python src/main.py --sim 1 --sim_type orthogonal_hard --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 4 --dim 40 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
#     --candidate_recursion 1 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 20 --train_percent 0.1 --thav 0 --thresh_const 0.8 --save_lambda 1
# python src/main.py --sim 1 --sim_type orthogonal_hard --single_test 0 --step_size 1 --window_size 150 --split_variance 0 --M 4 --dim 60 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
#     --candidate_recursion 1 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 20 --train_percent 0.1 --thav 0 --thresh_const 0.8 --save_lambda 1
# python src/main.py --sim 1 --sim_type orthogonal_hard --single_test 0 --step_size 1 --window_size 200 --split_variance 0 --M 4 --dim 80 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
#     --candidate_recursion 1 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 20 --train_percent 0.1 --thav 0 --thresh_const 0.8 --save_lambda 1

# python src/main.py --sim 1 --sim_type orthogonal_multiple_block --single_test 0 --step_size 1 --window_size 50 --split_variance 0 --M 4 --dim 20 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
#     --candidate_recursion 0 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 20 --train_percent 0.1 --thav 0 --thresh_const 0.8 --save_lambda 1
# python src/main.py --sim 1 --sim_type orthogonal_multiple_block --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 4 --dim 40 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
#     --candidate_recursion 0 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 20 --train_percent 0.1 --thav 0 --thresh_const 0.8 --save_lambda 1
python src/main.py --sim 1 --sim_type orthogonal_multiple_block --single_test 0 --step_size 1 --window_size 150 --split_variance 0 --M 4 --dim 60 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
    --candidate_recursion 0 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 20 --train_percent 0.1 --thav 0 --thresh_const 0.8 --save_lambda 1
python src/main.py --sim 1 --sim_type orthogonal_multiple_block --single_test 0 --step_size 1 --window_size 200 --split_variance 0 --M 4 --dim 80 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
    --candidate_recursion 0 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 20 --train_percent 0.1 --thav 0 --thresh_const 0.8 --save_lambda 1

# python src/main.py --sim 1 --sim_type orthogonal_cross_hard --single_test 0 --step_size 1 --window_size 50 --split_variance 0 --M 4 --dim 20 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
#     --candidate_recursion 1 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 20 --train_percent 0.1 --thav 0 --thresh_const 0.8 --save_lambda 1
# python src/main.py --sim 1 --sim_type orthogonal_cross_hard --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 4 --dim 40 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
#     --candidate_recursion 1 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 20 --train_percent 0.1 --thav 0 --thresh_const 0.8 --save_lambda 1
# python src/main.py --sim 1 --sim_type orthogonal_cross_hard --single_test 0 --step_size 1 --window_size 150 --split_variance 0 --M 4 --dim 60 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
#     --candidate_recursion 1 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 20 --train_percent 0.1 --thav 0 --thresh_const 0.8 --save_lambda 1
# python src/main.py --sim 1 --sim_type orthogonal_cross_hard --single_test 0 --step_size 1 --window_size 200 --split_variance 0 --M 4 --dim 80 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
#     --candidate_recursion 1 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 20 --train_percent 0.1 --thav 0 --thresh_const 0.8 --save_lambda 
    
python src/main.py --sim 1 --sim_type orthogonal_cross_block --single_test 0 --step_size 1 --window_size 50 --split_variance 0 --M 4 --dim 20 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
    --candidate_recursion 0 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 20 --train_percent 0.1 --thav 0 --thresh_const 0.8 --save_lambda 1
python src/main.py --sim 1 --sim_type orthogonal_cross_block --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 4 --dim 40 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
    --candidate_recursion 0 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 20 --train_percent 0.1 --thav 0 --thresh_const 0.8 --save_lambda 1
python src/main.py --sim 1 --sim_type orthogonal_cross_block --single_test 0 --step_size 1 --window_size 150 --split_variance 0 --M 4 --dim 60 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
    --candidate_recursion 0 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 20 --train_percent 0.1 --thav 0 --thresh_const 0.8 --save_lambda 1
python src/main.py --sim 1 --sim_type orthogonal_cross_block --single_test 0 --step_size 1 --window_size 200 --split_variance 0 --M 4 --dim 80 --optim_type Boyd --lam 5e-2 --N 500 --recursion 1 --linkage single \
    --candidate_recursion 0 --base_M 2 --sim_scale 0.8 --log_pvals 1 --post_window_size 20 --train_percent 0.1 --thav 0 --thresh_const 0.8 --save_lambda 1
