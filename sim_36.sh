#!/bin/bash
#python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 4 --sim_type cholesky --dim 20 --fix_pre 1
#python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 4 --sim_type cai_model_one --dim 20 --fix_pre 1
#python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 4 --sim_type cai_model_three --dim 20 --fix_pre 1
#python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 4 --sim_type orthogonal --dim 20 --fix_pre 1
# python src/main.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --single_test 0 --split_variance 1 --N 400 --M 6 --sim_type orthogonal --dim 36

# python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 4 --sim_type cholesky --dim 24 --fix_pre 1
# python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 4 --sim_type cai_model_one --dim 24 --fix_pre 1
# python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 4 --sim_type cai_model_three --dim 24 --fix_pre 1
# python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 4 --sim_type orthogonal --dim 24 --fix_pre 1

# python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 5 --sim_type cholesky --dim 30 --fix_pre 1
# python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 5 --sim_type cai_model_one --dim 30 --fix_pre 1
# python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 5 --sim_type cai_model_three --dim 30 --fix_pre 1
# python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 5 --sim_type orthogonal --dim 30 --fix_pre 1

# python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 6 --sim_type cholesky --dim 36 --fix_pre 1
# python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 6 --sim_type cai_model_one --dim 36 --fix_pre 1
# python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 6 --sim_type cai_model_three --dim 36 --fix_pre 1
# python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 6 --sim_type orthogonal --dim 36 --fix_pre 1


# python src/xia_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type unstructured --step_size 1 --window_size 100 --M 4 --dim 24 --lam 5e-2 --N 400
# python src/xia_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type unstructured --step_size 1 --window_size 100 --M 5 --dim 30 --lam 5e-2 --N 400
# python src/xia_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type unstructured --step_size 1 --window_size 100 --M 6 --dim 36 --lam 5e-2 --N 400
# python src/xia_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type unstructured --step_size 1 --window_size 100 --M 6 --dim 42 --lam 5e-2 --N 400
# python src/xia_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type unstructured --step_size 1 --window_size 100 --M 6 --dim 48 --lam 5e-2 --N 400
# python src/xia_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type unstructured --step_size 1 --window_size 100 --M 7 --dim 56 --lam 5e-2 --N 400
# python src/xia_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type unstructured --step_size 1 --window_size 100 --M 8 --dim 64 --lam 5e-2 --N 400

# python src/kesh_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type unstructured --step_size 1 --window_size 100 --M 4 --dim 24 --lam 5e-2 --N 400 --burn_in 100
# python src/kesh_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type unstructured --step_size 1 --window_size 100 --M 5 --dim 30 --lam 5e-2 --N 400 --burn_in 100
# python src/kesh_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type unstructured --step_size 1 --window_size 100 --M 6 --dim 36 --lam 5e-2 --N 400 --burn_in 100
# python src/kesh_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type unstructured --step_size 1 --window_size 100 --M 6 --dim 42 --lam 5e-2 --N 400 --burn_in 100
# python src/kesh_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type unstructured --step_size 1 --window_size 100 --M 6 --dim 48 --lam 5e-2 --N 400 --burn_in 100
# python src/kesh_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type unstructured --step_size 1 --window_size 100 --M 7 --dim 56 --lam 5e-2 --N 400 --burn_in 100
# python src/kesh_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type unstructured --step_size 1 --window_size 100 --M 8 --dim 64 --lam 5e-2 --N 400 --burn_in 100

# UNSTRUCTURED RESIDS
python src/main.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type unstructured --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 4 --dim 20 --optim_type Boyd --lam 5e-2 --N 400
python src/main.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type unstructured --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 5 --dim 40 --optim_type Boyd --lam 5e-2 --N 400
python src/main.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type unstructured --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 6 --dim 60 --optim_type Boyd --lam 5e-2 --N 400
python src/main.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type unstructured --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 8 --dim 80 --optim_type Boyd --lam 5e-2 --N 400

# BLOCK RESIDS
python src/main.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type block --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 4 --dim 20 --optim_type Boyd --lam 5e-2 --N 400
python src/main.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type block --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 5 --dim 40 --optim_type Boyd --lam 5e-2 --N 400
python src/main.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type block --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 6 --dim 60 --optim_type Boyd --lam 5e-2 --N 400
python src/main.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type block --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 8 --dim 80 --optim_type Boyd --lam 5e-2 --N 400

# CAI MODEL ONE
python src/main.py --sim 1 --sim_type cai_model_one --num_indices 4 --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 4 --dim 20 --optim_type Boyd --lam 5e-2 --N 400
python src/main.py --sim 1 --sim_type cai_model_one --num_indices 4 --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 5 --dim 40 --optim_type Boyd --lam 5e-2 --N 400
python src/main.py --sim 1 --sim_type cai_model_one --num_indices 4 --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 6 --dim 60 --optim_type Boyd --lam 5e-2 --N 400
python src/main.py --sim 1 --sim_type cai_model_one --num_indices 4 --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 8 --dim 80 --optim_type Boyd --lam 5e-2 --N 400

# CAI MODEL THREE
python src/main.py --sim 1 --sim_type cai_model_three --num_indices 4 --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 4 --dim 20 --optim_type Boyd --lam 5e-2 --N 400
python src/main.py --sim 1 --sim_type cai_model_three --num_indices 4 --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 5 --dim 40 --optim_type Boyd --lam 5e-2 --N 400
python src/main.py --sim 1 --sim_type cai_model_three --num_indices 4 --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 6 --dim 60 --optim_type Boyd --lam 5e-2 --N 400
python src/main.py --sim 1 --sim_type cai_model_three --num_indices 4 --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 8 --dim 80 --optim_type Boyd --lam 5e-2 --N 400

# Sparse Cholesky
python src/main.py --sim 1 --sim_type sparse_cholesky --num_coeffs_change 4 --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 4 --dim 20 --optim_type Boyd --lam 5e-2 --N 400
python src/main.py --sim 1 --sim_type sparse_cholesky --num_coeffs_change 4 --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 5 --dim 40 --optim_type Boyd --lam 5e-2 --N 400
python src/main.py --sim 1 --sim_type sparse_cholesky --num_coeffs_change 4 --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 6 --dim 60 --optim_type Boyd --lam 5e-2 --N 400
python src/main.py --sim 1 --sim_type sparse_cholesky --num_coeffs_change 4 --single_test 0 --step_size 1 --window_size 100 --split_variance 0 --M 8 --dim 80 --optim_type Boyd --lam 5e-2 --N 400

python src/xia_cpd.py --sim 1 --sim_type sparse_cholesky --num_coeffs_change 4 --step_size 1 --window_size 100 --M 4 --dim 20 --lam 5e-2 --N 400
python src/xia_cpd.py --sim 1 --sim_type sparse_cholesky --num_coeffs_change 4 --step_size 1 --window_size 100 --M 5 --dim 40 --lam 5e-2 --N 400
python src/xia_cpd.py --sim 1 --sim_type sparse_cholesky --num_coeffs_change 4 --step_size 1 --window_size 100 --M 6 --dim 60 --lam 5e-2 --N 400
python src/xia_cpd.py --sim 1 --sim_type sparse_cholesky --num_coeffs_change 4 --step_size 1 --window_size 100 --M 8 --dim 80 --lam 5e-2 --N 400

python src/kesh_cpd.py --sim 1 --sim_type sparse_cholesky --num_coeffs_change 4 --step_size 1 --window_size 100 --M 4 --dim 20 --lam 5e-2 --N 400 --burn_in 100
python src/kesh_cpd.py --sim 1 --sim_type sparse_cholesky --num_coeffs_change 4 --step_size 1 --window_size 100 --M 5 --dim 40 --lam 5e-2 --N 400 --burn_in 100
python src/kesh_cpd.py --sim 1 --sim_type sparse_cholesky --num_coeffs_change 4 --step_size 1 --window_size 100 --M 6 --dim 60 --lam 5e-2 --N 400 --burn_in 100
python src/kesh_cpd.py --sim 1 --sim_type sparse_cholesky --num_coeffs_change 4 --step_size 1 --window_size 100 --M 8 --dim 80 --lam 5e-2 --N 400 --burn_in 100


############################################################################

# python src/xia_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type unstructured --step_size 1 --window_size 100 --M 4 --dim 20 --lam 5e-2 --N 400
# python src/xia_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type unstructured --step_size 1 --window_size 100 --M 5 --dim 40 --lam 5e-2 --N 400
# python src/xia_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type unstructured --step_size 1 --window_size 100 --M 6 --dim 60 --lam 5e-2 --N 400
# python src/xia_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type unstructured --step_size 1 --window_size 100 --M 8 --dim 80 --lam 5e-2 --N 400

# python src/kesh_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type unstructured --step_size 1 --window_size 100 --M 4 --dim 20 --lam 5e-2 --N 400 --burn_in 100
# python src/kesh_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type unstructured --step_size 1 --window_size 100 --M 5 --dim 40 --lam 5e-2 --N 400 --burn_in 100
# python src/kesh_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type unstructured --step_size 1 --window_size 100 --M 6 --dim 60 --lam 5e-2 --N 400 --burn_in 100
# python src/kesh_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type unstructured --step_size 1 --window_size 100 --M 8 --dim 80 --lam 5e-2 --N 400 --burn_in 100


# python src/xia_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type block --step_size 1 --window_size 100 --M 4 --dim 20 --lam 5e-2 --N 400
# python src/xia_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type block --step_size 1 --window_size 100 --M 5 --dim 40 --lam 5e-2 --N 400
# python src/xia_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type block --step_size 1 --window_size 100 --M 6 --dim 60 --lam 5e-2 --N 400
# python src/xia_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type block --step_size 1 --window_size 100 --M 8 --dim 80 --lam 5e-2 --N 400

# python src/kesh_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type block --step_size 1 --window_size 100 --M 4 --dim 20 --lam 5e-2 --N 400 --burn_in 100
# python src/kesh_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type block --step_size 1 --window_size 100 --M 5 --dim 40 --lam 5e-2 --N 400 --burn_in 100
# python src/kesh_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type block --step_size 1 --window_size 100 --M 6 --dim 60 --lam 5e-2 --N 400 --burn_in 100
# python src/kesh_cpd.py --sim 1 --sim_type anderson_residual --num_indices 4 --resid_type block --step_size 1 --window_size 100 --M 8 --dim 80 --lam 5e-2 --N 400 --burn_in 100

##############################################################################
#python src/xia_cpd.py --sim 1 --sim_type cai_model_one --num_indices 4 --step_size 1 --window_size 100 --M 4 --dim 20 --lam 5e-2 --N 400
#python src/xia_cpd.py --sim 1 --sim_type cai_model_one --num_indices 4 --step_size 1 --window_size 100 --M 5 --dim 40 --lam 5e-2 --N 400
#python src/xia_cpd.py --sim 1 --sim_type cai_model_one --num_indices 4 --step_size 1 --window_size 100 --M 6 --dim 60 --lam 5e-2 --N 400
# python src/xia_cpd.py --sim 1 --sim_type cai_model_one --num_indices 4 --step_size 1 --window_size 100 --M 8 --dim 80 --lam 5e-2 --N 400

# python src/kesh_cpd.py --sim 1 --sim_type cai_model_one --num_indices 4 --step_size 1 --window_size 100 --M 4 --dim 20 --lam 5e-2 --N 400 --burn_in 100
# python src/kesh_cpd.py --sim 1 --sim_type cai_model_one --num_indices 4 --step_size 1 --window_size 100 --M 5 --dim 40 --lam 5e-2 --N 400 --burn_in 100
# python src/kesh_cpd.py --sim 1 --sim_type cai_model_one --num_indices 4 --step_size 1 --window_size 100 --M 6 --dim 60 --lam 5e-2 --N 400 --burn_in 100
# python src/kesh_cpd.py --sim 1 --sim_type cai_model_one --num_indices 4 --step_size 1 --window_size 100 --M 8 --dim 80 --lam 5e-2 --N 400 --burn_in 100


# python src/xia_cpd.py --sim 1 --sim_type cai_model_three --num_indices 4 --step_size 1 --window_size 100 --M 4 --dim 20 --lam 5e-2 --N 400
# python src/xia_cpd.py --sim 1 --sim_type cai_model_three --num_indices 4 --step_size 1 --window_size 100 --M 5 --dim 40 --lam 5e-2 --N 400
# python src/xia_cpd.py --sim 1 --sim_type cai_model_three --num_indices 4 --step_size 1 --window_size 100 --M 6 --dim 60 --lam 5e-2 --N 400
# python src/xia_cpd.py --sim 1 --sim_type cai_model_three --num_indices 4 --step_size 1 --window_size 100 --M 8 --dim 80 --lam 5e-2 --N 400

# python src/kesh_cpd.py --sim 1 --sim_type cai_model_three --num_indices 4 --step_size 1 --window_size 100 --M 4 --dim 20 --lam 5e-2 --N 400 --burn_in 100
# python src/kesh_cpd.py --sim 1 --sim_type cai_model_three --num_indices 4 --step_size 1 --window_size 100 --M 5 --dim 40 --lam 5e-2 --N 400 --burn_in 100
# python src/kesh_cpd.py --sim 1 --sim_type cai_model_three --num_indices 4 --step_size 1 --window_size 100 --M 6 --dim 60 --lam 5e-2 --N 400 --burn_in 100
# python src/kesh_cpd.py --sim 1 --sim_type cai_model_three --num_indices 4 --step_size 1 --window_size 100 --M 8 --dim 80 --lam 5e-2 --N 400 --burn_in 100


# dims 20, 40, 80, optionally (100)
# andersons + sparse cholesky




# 


