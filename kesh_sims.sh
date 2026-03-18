#!/bin/bash

python src/kesh_cpd.py --sim 1 --sim_type orthogonal_small --step_size 1 --window_size 50 --M 4 --dim 20 --N 500 --burn_in 50 --sim_scale 0.8 --post_window_size 20 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type orthogonal_small --step_size 1 --window_size 100 --M 4 --dim 40 --N 500 --burn_in 100 --sim_scale 0.8 --post_window_size 20 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type orthogonal_small --step_size 1 --window_size 150 --M 4 --dim 60 --N 500 --burn_in 150 --sim_scale 0.8 --post_window_size 20 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type orthogonal_small --step_size 1 --window_size 200 --M 4 --dim 80 --N 500 --burn_in 200 --sim_scale 0.8 --post_window_size 20 --estimator clime --auto_lambda 1

python src/kesh_cpd.py --sim 1 --sim_type orthogonal_small --step_size 1 --window_size 50 --M 4 --dim 20 --N 500 --burn_in 50 --sim_scale 0.8 --post_window_size 20 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type orthogonal_small --step_size 1 --window_size 100 --M 4 --dim 40 --N 500 --burn_in 100 --sim_scale 0.8 --post_window_size 20 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type orthogonal_small --step_size 1 --window_size 150 --M 4 --dim 60 --N 500 --burn_in 150 --sim_scale 0.8 --post_window_size 20 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type orthogonal_small --step_size 1 --window_size 200 --M 4 --dim 80 --N 500 --burn_in 200 --sim_scale 0.8 --post_window_size 20 --alt 1 --estimator clime --auto_lambda 1

python src/kesh_cpd.py --sim 1 --sim_type orthogonal_cross_block --step_size 1 --window_size 50 --M 4 --dim 20 --N 500 --burn_in 50 --sim_scale 0.8 --post_window_size 20 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type orthogonal_cross_block --step_size 1 --window_size 100 --M 4 --dim 40 --N 500 --burn_in 100 --sim_scale 0.8 --post_window_size 20 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type orthogonal_cross_block --step_size 1 --window_size 150 --M 4 --dim 60 --N 500 --burn_in 150 --sim_scale 0.8 --post_window_size 20 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type orthogonal_cross_block --step_size 1 --window_size 200 --M 4 --dim 80 --N 500 --burn_in 200 --sim_scale 0.8 --post_window_size 20 --estimator clime --auto_lambda 1

python src/kesh_cpd.py --sim 1 --sim_type orthogonal_cross_block --step_size 1 --window_size 50 --M 4 --dim 20 --N 500 --burn_in 50 --sim_scale 0.8 --post_window_size 20 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type orthogonal_cross_block --step_size 1 --window_size 100 --M 4 --dim 40 --N 500 --burn_in 100 --sim_scale 0.8 --post_window_size 20 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type orthogonal_cross_block --step_size 1 --window_size 150 --M 4 --dim 60 --N 500 --burn_in 150 --sim_scale 0.8 --post_window_size 20 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type orthogonal_cross_block --step_size 1 --window_size 200 --M 4 --dim 80 --N 500 --burn_in 200 --sim_scale 0.8 --post_window_size 20 --alt 1 --estimator clime --auto_lambda 1

python src/kesh_cpd.py --sim 1 --sim_type orthogonal_multiple_block --step_size 1 --window_size 50 --M 4 --dim 20 --N 500 --burn_in 50 --sim_scale 0.8 --post_window_size 20 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type orthogonal_multiple_block --step_size 1 --window_size 100 --M 4 --dim 40 --N 500 --burn_in 100 --sim_scale 0.8 --post_window_size 20 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type orthogonal_multiple_block --step_size 1 --window_size 150 --M 4 --dim 60 --N 500 --burn_in 150 --sim_scale 0.8 --post_window_size 20 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type orthogonal_multiple_block --step_size 1 --window_size 200 --M 4 --dim 80 --N 500 --burn_in 200 --sim_scale 0.8 --post_window_size 20 --estimator clime --auto_lambda 1

python src/kesh_cpd.py --sim 1 --sim_type orthogonal_multiple_block --step_size 1 --window_size 50 --M 4 --dim 20 --N 500 --burn_in 50 --sim_scale 0.8 --post_window_size 20 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type orthogonal_multiple_block --step_size 1 --window_size 100 --M 4 --dim 40 --N 500 --burn_in 100 --sim_scale 0.8 --post_window_size 20 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type orthogonal_multiple_block --step_size 1 --window_size 150 --M 4 --dim 60 --N 500 --burn_in 150 --sim_scale 0.8 --post_window_size 20 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type orthogonal_multiple_block --step_size 1 --window_size 200 --M 4 --dim 80 --N 500 --burn_in 200 --sim_scale 0.8 --post_window_size 20 --alt 1 --estimator clime --auto_lambda 1

python src/kesh_cpd.py --sim 1 --sim_type orthogonal_hard --step_size 1 --window_size 50 --M 4 --dim 20 --N 500 --burn_in 50 --sim_scale 0.8 --post_window_size 20 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type orthogonal_hard --step_size 1 --window_size 100 --M 4 --dim 40 --N 500 --burn_in 100 --sim_scale 0.8 --post_window_size 20 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type orthogonal_hard --step_size 1 --window_size 150 --M 4 --dim 60 --N 500 --burn_in 150 --sim_scale 0.8 --post_window_size 20 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type orthogonal_hard --step_size 1 --window_size 200 --M 4 --dim 80 --N 500 --burn_in 200 --sim_scale 0.8 --post_window_size 20 --estimator clime --auto_lambda 1

python src/kesh_cpd.py --sim 1 --sim_type orthogonal_hard --step_size 1 --window_size 50 --M 4 --dim 20 --N 500 --burn_in 50 --sim_scale 0.8 --post_window_size 20 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type orthogonal_hard --step_size 1 --window_size 100 --M 4 --dim 40 --N 500 --burn_in 100 --sim_scale 0.8 --post_window_size 20 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type orthogonal_hard --step_size 1 --window_size 150 --M 4 --dim 60 --N 500 --burn_in 150 --sim_scale 0.8 --post_window_size 20 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type orthogonal_hard --step_size 1 --window_size 200 --M 4 --dim 80 --N 500 --burn_in 200 --sim_scale 0.8 --post_window_size 20 --alt 1 --estimator clime --auto_lambda 1

python src/kesh_cpd.py --sim 1 --sim_type cai_model_one --step_size 1 --window_size 50 --dim 20 --N 500 --burn_in 50 --post_window_size 20 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type cai_model_one --step_size 1 --window_size 100 --dim 40 --N 500 --burn_in 100 --post_window_size 20 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type cai_model_one --step_size 1 --window_size 150 --dim 60 --N 500 --burn_in 150 --post_window_size 20 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type cai_model_one --step_size 1 --window_size 200 --dim 80 --N 500 --burn_in 200 --post_window_size 20 --estimator clime --auto_lambda 1

python src/kesh_cpd.py --sim 1 --sim_type cai_model_one --step_size 1 --window_size 50 --dim 20 --N 500 --burn_in 50 --post_window_size 20 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type cai_model_one --step_size 1 --window_size 100 --dim 40 --N 500 --burn_in 100 --post_window_size 20 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type cai_model_one --step_size 1 --window_size 150 --dim 60 --N 500 --burn_in 150 --post_window_size 20 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type cai_model_one --step_size 1 --window_size 200 --dim 80 --N 500 --burn_in 200 --post_window_size 20 --alt 1 --estimator clime --auto_lambda 1

python src/kesh_cpd.py --sim 1 --sim_type cai_model_three --step_size 1 --window_size 50 --dim 20 --N 500 --burn_in 50 --post_window_size 20 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type cai_model_three --step_size 1 --window_size 100 --dim 40 --N 500 --burn_in 100 --post_window_size 20 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type cai_model_three --step_size 1 --window_size 150 --dim 60 --N 500 --burn_in 150 --post_window_size 20 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type cai_model_three --step_size 1 --window_size 200 --dim 80 --N 500 --burn_in 200 --post_window_size 20 --estimator clime --auto_lambda 1

python src/kesh_cpd.py --sim 1 --sim_type cai_model_three --step_size 1 --window_size 50 --dim 20 --N 500 --burn_in 50 --post_window_size 20 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type cai_model_three --step_size 1 --window_size 100 --dim 40 --N 500 --burn_in 100 --post_window_size 20 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type cai_model_three --step_size 1 --window_size 150 --dim 60 --N 500 --burn_in 150 --post_window_size 20 --alt 1 --estimator clime --auto_lambda 1
python src/kesh_cpd.py --sim 1 --sim_type cai_model_three --step_size 1 --window_size 200 --dim 80 --N 500 --burn_in 200 --post_window_size 20 --alt 1 --estimator clime --auto_lambda 1

