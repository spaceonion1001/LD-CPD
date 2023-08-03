#!/bin/bash
#python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 4 --sim_type cholesky --dim 20 --fix_pre 1
#python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 4 --sim_type cai_model_one --dim 20 --fix_pre 1
#python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 4 --sim_type cai_model_three --dim 20 --fix_pre 1
#python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 4 --sim_type orthogonal --dim 20 --fix_pre 1
python src/main.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --single_test 0 --split_variance 1 --N 400 --M 6 --sim_type orthogonal --dim 36

python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 4 --sim_type cholesky --dim 24 --fix_pre 1
python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 4 --sim_type cai_model_one --dim 24 --fix_pre 1
python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 4 --sim_type cai_model_three --dim 24 --fix_pre 1
python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 4 --sim_type orthogonal --dim 24 --fix_pre 1

python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 5 --sim_type cholesky --dim 30 --fix_pre 1
python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 5 --sim_type cai_model_one --dim 30 --fix_pre 1
python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 5 --sim_type cai_model_three --dim 30 --fix_pre 1
python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 5 --sim_type orthogonal --dim 30 --fix_pre 1

python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 6 --sim_type cholesky --dim 36 --fix_pre 1
python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 6 --sim_type cai_model_one --dim 36 --fix_pre 1
python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 6 --sim_type cai_model_three --dim 36 --fix_pre 1
python src/xia_cpd.py --sim 1 --window_size 100 --step_size 1 --lam 8e-2 --N 400 --M 6 --sim_type orthogonal --dim 36 --fix_pre 1


