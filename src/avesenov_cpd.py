from pickletools import read_string1
from cvxpy import vec
import numpy as np
from numpy.linalg import inv as inv
from scipy.optimize import least_squares, minimize
from simulate import sim_changepoint_mv_normal_cholesky, sim_changepoint_mv_normal_ldlt, sim_changepoint_var_process, changepoint_cai_model_one, changepoint_cai_model_three, anderson_sim_with_residual
from utils import scale_data, difference_data, vectorize_matrix, symmetrize_from_vector
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm
from scipy.stats import norm
from simulate import *
from utils import difference_data, load_alaska_data, scale_data, load_hjandrews_data, create_fig_dir, load_holiday_farm_data, load_tohoku_data, load_stock_market_data, load_mesonet_data
from numba import jit
import time
from datetime import timedelta
from scipy.optimize import linprog
import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages

import matplotlib.pyplot as plt
import matplotlib

import argparse

matplotlib.use("Agg")

r = robjects.r
rpy2.robjects.numpy2ri.activate()
utils = importr('utils')
utils.chooseCRANmirror(ind=1)

covcp = importr('covcp')
glasso = importr('glasso')
parallel = importr('parallel')

def resolve_data(args, save_path=None, data_seed=42):
    if bool(args.sim):
        if args.sim_type == 'orthogonal_small':
            return sim_changepoint_mv_normal_orthogonal(sim_scale=args.sim_scale, M=args.M, dim=args.dim, N=args.N, save_path=save_path, data_seed=data_seed)[1].T
        elif args.sim_type == 'orthogonal_mult_coeff':
            return sim_changepoint_mv_normal_orthogonal_mult_coeff(sim_scale=args.sim_scale, num_coeffs_change=args.num_coeffs_change, M=args.M, dim=args.dim, N=args.N, save_path=save_path)[1].T
        elif args.sim_type == 'cholesky':
            return scale_data(sim_changepoint_mv_normal_cholesky(dim=args.dim, N=args.N, num_coeffs_change=args.num_coeffs_change, scale=args.sim_scale, save_path=save_path))
        elif args.sim_type == 'ldlt':
            return scale_data(sim_changepoint_mv_normal_ldlt(dim=args.dim, N=args.N, num_coeffs_change=args.num_coeffs_change, scale=args.sim_scale, save_path=save_path))
        elif args.sim_type == 'var_process':
            return scale_data(difference_data(sim_changepoint_var_process(dim=args.dim, N=args.N, num_coeffs_change=args.num_coeffs_change, scale=args.sim_scale, save_path=save_path)))
        elif args.sim_type == 'cai_model_one':
            return scale_data(changepoint_cai_model_one(args, dim=args.dim, N=args.N, save_path=save_path))
        elif args.sim_type == 'cai_model_three':
            return scale_data(changepoint_cai_model_three(args, dim=args.dim, N=args.N, save_path=save_path))
        elif args.sim_type == 'orthogonal_no_change':
            return sim_changepoint_mv_normal_orthogonal_no_change(sim_scale=args.sim_scale, M=args.M, dim=args.dim, N=args.N, save_path=save_path)[1].T
        elif args.sim_type == 'cholesky_no_change':
            return scale_data(sim_changepoint_mv_normal_cholesky_no_change(dim=args.dim, N=args.N, num_coeffs_change=args.num_coeffs_change, scale=args.sim_scale, save_path=save_path))
        elif args.sim_type == 'cai_model_one_no_change':
            return scale_data(changepoint_cai_model_one_no_change(dim=args.dim, N=args.N, save_path=save_path))
        elif args.sim_type == 'cai_model_three_no_change':
            return scale_data(changepoint_cai_model_three_no_change(dim=args.dim, N=args.N, save_path=save_path))
        elif args.sim_type == 'sparse_cholesky':
            return scale_data(sim_changepoint_mv_normal_cholesky(dim=args.dim, N=args.N, num_coeffs_change=args.num_coeffs_change, scale=args.sim_scale, save_path=save_path, sparse=True))
        elif args.sim_type == 'sparse_cholesky_no_change':
            return scale_data(sim_changepoint_mv_normal_cholesky_no_change(dim=args.dim, N=args.N, num_coeffs_change=args.num_coeffs_change, scale=args.sim_scale, save_path=save_path, sparse=True))
        elif args.sim_type == 'cai_model_four':
            return scale_data(changepoint_cai_model_four(dim=args.dim, N=args.N, save_path=save_path))
        elif args.sim_type == 'cai_model_four_no_change':
            return scale_data(changepoint_cai_model_four_no_change(dim=args.dim, N=args.N, save_path=save_path))
        elif args.sim_type == 'anderson_residual':
            return scale_data(anderson_sim_with_residual(M=args.M, dim=args.dim, N=args.N, num_indices=args.num_indices, resid_type=args.resid_type, save_path=save_path), args.train_percent)
        else:
            print("Incorrect Simulation")
            exit(0)
    else:
        if args.data == 'alaska':
            return load_alaska_data(args)
        elif args.data == 'tohoku':
            return scale_data(load_tohoku_data(args))
        elif args.data == 'hjandrews':
            return load_hjandrews_data(args)
        elif args.data == 'holidayfarm':
            return load_holiday_farm_data(args)
        elif args.data == 'stocks':
            # python src/main.py --M 5 --lam 2e-5 --window_size 200 --step_size 1 --data stocks --local 1 --sim 0 --split_variance 0 --train_percent 0.6
            # stocks need a low lambda value I think
            return scale_data(load_stock_market_data(args), args.train_percent)
        elif args.data == 'mesonet':
            return scale_data(load_mesonet_data(args), args.train_percent)
        else:
            print("Error: Dataset not understood")
            exit(0)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lam', type=float, default=1e-1)
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--N', type=int, default=500)
    parser.add_argument('--M', type=int, default=2)
    parser.add_argument('--sim', type=int, default=1)
    parser.add_argument('--sim_type', type=str, default='cai_model_three')
    parser.add_argument('--data', type=str, default='alaska')
    parser.add_argument('--data_path', type=str, default='/home/dink/Documents/Research/data')
    parser.add_argument('--data_fname', type=str, default='mesonet_test_out.csv')
    parser.add_argument('--sim_scale', type=float, default=0.8)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--train_percent', type=float, default=0.25)
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--post_window_size', type=int, default=20)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--num_coeffs_change', type=int, default=2)
    parser.add_argument('--results_path', type=str, default='/home/dink/Documents/Research/Correlation-Changepoint-Detection/results')
    parser.add_argument('--fix_pre', type=int, default=1)
    parser.add_argument('--resid_type', type=str, choices=['unstructured', 'block'], default='unstructured', help='Residual Type')
    parser.add_argument('--num_indices', type=int, default=4)
    parser.add_argument('--single_test', type=int, default=0)
    parser.add_argument('--results_fldr_name', type=str, default=None)
    parser.add_argument('--results_filename', type=str, default=None)
    args = parser.parse_args()

    return args

def run_avesenov(args, data):
    data_one = data[0:args.window_size, :]
    C_train = np.cov(data_one.T)
    r_max = np.max(np.abs(C_train[np.triu_indices(C_train.shape[0], k=1)]))
    lambda_search = [0.05 + (j*(r_max-0.05))/40 for j in range(1, 41, 1)]
    glasso_one = GraphicalLassoCV(alphas=lambda_search, n_refinements=4, tol=1e-4, max_iter=1500, cv=5).fit(data_one)
    precision_one = glasso_one.precision_
    chosen_lamb = glasso_one.alpha_
    T_n_one = precision_one + precision_one.T - (precision_one.T @ C_train @ precision_one)

    for i in tqdm(range(args.window_size, data.shape[0]-args.post_window_size, args.step_size)):
        data_two = data[i:i+args.post_window_size, :]
        C_two = np.cov(data_two.T)
        glasso_two = GraphicalLasso(alpha=chosen_lamb, tol=1e-4, max_iter=1500).fit(data_two)
        precision_two = glasso_two.precision_
        T_n_two = precision_two + precision_two.T - (precision_one.T @ C_two @ precision_two)


def run_avesenov_R(args, data):
    windows = [args.window_size]
    alpha = robjects.FloatVector([0.05])
    nrow, ncol = data.shape
    X = r.matrix(data, nrow=nrow, ncol=ncol)
    windows = robjects.IntVector(windows)
    stable_set = robjects.IntVector(np.arange(0, args.window_size))
    gl_func_code = """
    function(data) glasso(myCov(data), chooseRho(data), penalize.diagonal = F)$wi 
    """
    inf_norm_code = """
    infNorm = function(x) max(abs(x))
    """
    no_pattern_code = """
    noPattern = function(x) max(x$distances)
    """
    choose_rho_code = """
    chooseRho = function(data) sqrt(log(ncol(data)) / nrow(data))
    """
    cov_code = """
    myCov = function(X) t(X) %*% X/nrow(X)
    """
    my_cov_func = robjects.r(cov_code)
    choose_rho_func = robjects.r(choose_rho_code)
    no_pattern_func = robjects.r(no_pattern_code)
    inf_norm_func = robjects.r(inf_norm_code)
    GL = robjects.r(gl_func_code)


    test_result_R = covcp.createPrecisionMatrixTest(
        windows,
        alpha,
        X,
        covcp.noPattern,
        GL,
        covcp.infNorm,
        stable_set,
        covcp.hatTheta2GaussianSigmas
    )
    test_stats = np.array(test_result_R.rx2('statistics').rx2('window2statistics')[0].rx2('distances'))
    

    return test_stats
    
    # soln_dict = dict(zip(test_result_R.names, list(test_result_R)))
    # print(soln_dict['window2statistics'])

def perform_single_run(args):
    data_full = resolve_data(args, save_path=None)
    print(data_full.shape)
    save_path = os.path.join(args.results_path+"_avesenov", args.data)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    test_stats = run_avesenov_R(args, data=data_full)
    np.savetxt(os.path.join(save_path, args.data_fname+"_global_test_vals.csv"), test_stats, delimiter=',')
    plt.plot(test_stats)
    plt.savefig(os.path.join(save_path, args.data_fname+".png"))
    plt.close()

def perform_simulation_batch(args):
    print("\n*******************************************************************************")
    print("Performing Batch Simulation of {} with Dim = {}, Window = {}".format(args.sim_type, args.dim, args.window_size))
    seeds_list = np.arange(50, 70)
    sim_results_path = os.path.join(args.results_path, "simulation_results_avesenov")
    if not os.path.isdir(sim_results_path):
        os.mkdir(sim_results_path)
    sim_type_path = os.path.join(sim_results_path, args.sim_type+"_"+str(args.dim))
    if args.sim_type == 'anderson_residual':
        sim_type_path = os.path.join(sim_results_path, args.sim_type+"_"+args.resid_type+"_"+str(args.dim))
    print(sim_type_path)
    if not os.path.isdir(sim_type_path):
        os.mkdir(sim_type_path)
    for seed in seeds_list:
        np.random.seed(seed)
        save_path = os.path.join(sim_type_path, str(seed))
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        data_full = resolve_data(args, save_path=save_path, data_seed=seed)
        print(data_full.shape)
        global_test_vals = run_avesenov_R(args, data=data_full)
        print("Test Vals Shape", global_test_vals.shape)
        print()
        np.savetxt(os.path.join(save_path, "global_test_vals.csv"), global_test_vals, delimiter=',')
        plt.plot(global_test_vals)
        plt.savefig(os.path.join(save_path, "test_stat.png"))
        plt.close()
    print("*******************************************************************************")
    print("Done!")



if __name__ == '__main__':
    #main()
    args = get_args()
    if args.single_test:
        perform_single_run(args)
    else:
        perform_simulation_batch(args)