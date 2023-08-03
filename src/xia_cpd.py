from pickletools import read_string1
from cvxpy import vec
import numpy as np
from numpy.linalg import inv as inv
from scipy.optimize import least_squares, minimize
from simulate import sim_changepoint_mv_normal_cholesky, sim_changepoint_mv_normal_ldlt, sim_changepoint_var_process, changepoint_cai_model_one, changepoint_cai_model_three
from utils import scale_data, difference_data, vectorize_matrix, symmetrize_from_vector
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm
from scipy.stats import norm
from simulate import *
from utils import difference_data, load_alaska_data, scale_data, load_hjandrews_data, create_fig_dir, load_holiday_farm_data, load_tohoku_data, load_stock_market_data
from numba import jit
import time
from datetime import timedelta
from scipy.optimize import linprog
import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages

r = robjects.r
rpy2.robjects.numpy2ri.activate()
utils = importr('utils')
utils.chooseCRANmirror(ind=1)
# R package names
#packnames = ('fastclime', 'scalreg')
#packnames = ('scalreg', 'hexbin')

# R vector of strings
from rpy2.robjects.vectors import StrVector

# Selectively install what needs to be install.
# We are fancy, just because we can.
#names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
# if len(names_to_install) > 0:
#     utils.install_packages(StrVector(names_to_install))
#clime = importr('fastclime')
scalreg = importr('scalreg')

import argparse

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
    parser.add_argument('--sim_scale', type=float, default=0.8)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--train_percent', type=float, default=0.25)
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--num_coeffs_change', type=int, default=2)
    parser.add_argument('--results_path', type=str, default='/home/dink/Documents/Research/Correlation-Changepoint-Detection/results')
    parser.add_argument('--fix_pre', type=int, default=1)
    args = parser.parse_args()

    return args


def resolve_data(args, save_path=None):
    if bool(args.sim):
        if args.sim_type == 'orthogonal':
            return sim_changepoint_mv_normal_orthogonal(sim_scale=args.sim_scale, M=args.M, dim=args.dim, N=args.N, save_path=save_path)[1].T
        elif args.sim_type == 'orthogonal_mult_coeff':
            return sim_changepoint_mv_normal_orthogonal_mult_coeff(sim_scale=args.sim_scale, num_coeffs_change=args.num_coeffs_change, M=args.M, dim=args.dim, N=args.N, save_path=save_path)[1].T
        elif args.sim_type == 'cholesky':
            return scale_data(sim_changepoint_mv_normal_cholesky(dim=args.dim, N=args.N, num_coeffs_change=args.num_coeffs_change, scale=args.sim_scale, save_path=save_path))
        elif args.sim_type == 'ldlt':
            return scale_data(sim_changepoint_mv_normal_ldlt(dim=args.dim, N=args.N, num_coeffs_change=args.num_coeffs_change, scale=args.sim_scale, save_path=save_path))
        elif args.sim_type == 'var_process':
            return scale_data(difference_data(sim_changepoint_var_process(dim=args.dim, N=args.N, num_coeffs_change=args.num_coeffs_change, scale=args.sim_scale, save_path=save_path)))
        elif args.sim_type == 'cai_model_one':
            return scale_data(changepoint_cai_model_one(dim=args.dim, N=args.N, save_path=save_path))
        elif args.sim_type == 'cai_model_three':
            return scale_data(changepoint_cai_model_three(dim=args.dim, N=args.N, save_path=save_path))
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
        else:
            print("Error: Dataset not understood")
            exit(0)

def loss_func(x, data, i=0, lam=1e-4):
    N = data.shape[0]
    X_m_i = np.delete(data, i, axis=1)
    X_i = data[:, i]
    D = np.power(np.diag(np.cov(X_m_i.T, bias=False)), -0.5)
    mu_m_i = np.mean(X_m_i.T, 1)
    mu_i = np.mean(X_i)
    X_m_i_min_mu = X_m_i - mu_m_i
    X_i_min_mu = X_i - mu_i
    const = 1/(2*N)
    pred = X_m_i_min_mu@(D*x)
    func = const*(np.sum((pred - X_i_min_mu)**2))
    l1_reg = lam*np.linalg.norm(x, ord=1)

    return func+l1_reg


def perform_regression(data, kd=2):
    """
    Split data in half and perform the regression process on each column
    """
    half = data.shape[0]//2
    X_data = data[:half, :]
    Y_data = data[half:, :]
    dim = X_data.shape[1]
    x_0 = np.random.normal(0, 1, dim-1)
    x_log_dim = np.log(dim)/X_data.shape[0]
    y_log_dim = np.log(dim)/Y_data.shape[0]
    beta_hats_x = []
    beta_hats_y = []
    residuals_x = np.zeros((X_data.shape[0], dim))
    residuals_y = np.zeros((Y_data.shape[0], dim))
    beta_hats_x = np.zeros((dim, dim-1))
    beta_hats_y = np.zeros((dim, dim-1))
    for i in range(dim):
        X_m_i = np.delete(X_data, i, axis=1)
        y_i = X_data[:, i]
        sig_i_i_x = np.var(y_i)
        x_lam = kd*np.power(sig_i_i_x*x_log_dim, 0.5)
        nrow, ncol = X_m_i.shape
        X = r.matrix(X_m_i, nrow=nrow, ncol=ncol)
        y_i = robjects.FloatVector(X_data[:, i])
        reg_soln = scalreg.scalreg(X, y_i, lam0=x_lam)
        reg_soln_dict = dict(zip(reg_soln.names, list(reg_soln)))
        residuals_x[:, i] = np.array(reg_soln_dict['residuals'])
        beta_hats_x_curr = np.array(reg_soln_dict['coefficients'])
        beta_hats_x[i, :] = beta_hats_x_curr

        Y_m_i = np.delete(Y_data, i, axis=1)
        yy_i = Y_data[:, i]
        sig_i_i_y = np.var(yy_i)
        y_lam = kd*np.power(sig_i_i_y*y_log_dim, 0.5)
        nrow, ncol = Y_m_i.shape
        Y = r.matrix(Y_m_i, nrow=nrow, ncol=ncol)
        yy_i = robjects.FloatVector(Y_data[:, i])
        reg_soln = scalreg.scalreg(Y, yy_i, lam0=y_lam)
        reg_soln_dict = dict(zip(reg_soln.names, list(reg_soln)))
        residuals_y[:, i] = np.array(reg_soln_dict['residuals'])
        beta_hats_y_curr = np.array(reg_soln_dict['coefficients'])
        beta_hats_y[i, :] = beta_hats_y_curr

    return residuals_x, residuals_y, beta_hats_x, beta_hats_y


    #print(X_data.shape, r.dim(X))
    #print(scalreg.scalreg(X))
    # for i in range(dim):
    #     X_m_i = np.delete(X_data, i, axis=1)
    #     X_i = X_data[:, i]
    #     sig_i_i_x = np.var(X_i)
    #     x_lam = kd*np.power(sig_i_i_x*x_log_dim, 0.5)
    #     D_i_x = np.power(np.diag(np.cov(X_m_i.T, bias=False)), -0.5)
    #     res_x_i = minimize(loss_func, x_0, method='Nelder-Mead', args=(X_data, i, x_lam)).x
    #     #res_x_i = least_squares(loss_func, x_0, args=(X_data, i, x_lam)).x
    #     beta_hat_x_i = D_i_x*res_x_i

    #     Y_m_i = np.delete(Y_data, i, axis=1)
    #     Y_i = Y_data[:, i]
    #     sig_i_i_y = np.var(Y_i)
    #     y_lam = kd*np.power(sig_i_i_y*y_log_dim, 0.5)
    #     D_i_y = np.power(np.diag(np.cov(Y_m_i.T, bias=False)), -0.5)
    #     #res_y_i = minimize(loss_func, x_0, method='BFGS', args=(Y_data, i, y_lam)).x
    #     res_y_i = least_squares(loss_func, x_0, args=(Y_data, i, y_lam)).x
    #     beta_hat_y_i = D_i_y*res_y_i

    #     beta_hats_x.append(beta_hat_x_i)
    #     beta_hats_y.append(beta_hat_y_i)
    
    # return should be (dim, dim-1) shape
    # since we need one regression for each dim, and is vector size of dim-1
    #return np.array(beta_hats_x), np.array(beta_hats_y)

@jit(nopython=True)
def numba_mean_col(X):
    X_mean = []
    tot = X.shape[1]
    for col in range(X.shape[0]):
        sum = 0
        curr_data = X[:, col]
        for val in curr_data:
            sum = sum + val
        X_mean.append(sum/tot)
    X_mean = np.array(X_mean)
    return X_mean

@jit(nopython=True)
def calculate_residuals(beta_hats, data):
    # this is for one half of the data
    N = data.shape[0]
    dim = data.shape[1]
    residuals = np.zeros((N, dim))
    for k in range(N):
        for i in range(dim):
            #X_m_i = np.delete(data, i, axis=1)
            idxs = []
            for j in range(dim):
                if j != i:
                    idxs.append(int(j))
            idxs = np.array(idxs)
            X_m_i = data[:, idxs]
            X_m_i_k = X_m_i[k, :]
            X_i_k = data[k, i]
            # X_m_i_k_min_mu = X_m_i_k - np.mean(X_m_i.T, 1)
            # X_i_k_min_mu = X_i_k - np.mean(data[:, i])
            X_m_i_k_min_mu = X_m_i_k - numba_mean_col(X_m_i.T)
            X_i_k_min_mu = X_i_k - np.mean(data[:, i])
            pred = X_m_i_k_min_mu.dot(beta_hats[i])
            epsilon = X_i_k_min_mu - pred
            residuals[k, i] = epsilon
    
    return residuals

@jit(nopython=True)
def bias_corrected_residual_covariance(residuals, beta_hats):
    residuals_cov = np.cov(residuals.T, bias=True)
    dim = residuals_cov.shape[0]
    residuals_cov_corrected = np.zeros(residuals_cov.shape)
    for i in range(dim-1):
        for j in range(i+1, dim):
            first = residuals_cov[i, j]
            second = residuals_cov[i, i]*beta_hats.T[i, j]
            third = residuals_cov[j, j]*beta_hats.T[j-1, i]
            residuals_cov_corrected[i, j] = -(first+second+third)
            residuals_cov_corrected[j, i] = residuals_cov_corrected[i, j]
    
    return residuals_cov_corrected + np.diag(np.diag(residuals_cov))

@jit(nopython=True)
def calculate_T(residuals_cov_corrected):
    dim = residuals_cov_corrected.shape[0]
    T = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            top = residuals_cov_corrected[i, j]
            bottom = residuals_cov_corrected[i,i]*residuals_cov_corrected[j,j]
            T[i, j] = top/bottom
    
    return T

@jit(nopython=True)
def calculate_theta(residuals_cov_corrected, beta_hats, N):
    dim = residuals_cov_corrected.shape[0]
    theta = np.zeros((dim, dim))

    for i in range(dim-1):
        for j in range(i+1, dim):
            top = (1 + (beta_hats.T[i, j]**2)*residuals_cov_corrected[i, i]/residuals_cov_corrected[j,j])
            bottom = N*residuals_cov_corrected[i,i]*residuals_cov_corrected[j,j]
            theta[i,j] = top/bottom
            theta[j,i] = theta[i,j]
    
    np.fill_diagonal(theta, np.diag(residuals_cov_corrected))
    return theta

@jit(nopython=True)
def calculate_standardized_stat(T_x, T_y, theta_x, theta_y):
    W = (T_x - T_y)/np.sqrt(theta_x+theta_y)

    return W

@jit(nopython=True)
def calculate_global_stat(W):
    M = np.max(W**2)

    return M

@jit(nopython=True)
def indicator_global_stat(M, p, alpha=0.01):
    q_alpha = -np.log(8*np.pi)-2*np.log(np.log(np.power(1-alpha, -1)))
    threshold = q_alpha+4*np.log(p)-np.log(np.log(p))

    return threshold

def G_func(t):
    return 2-2*norm.cdf(t)

def indicator_local_stat(W, p, alpha=0.01):
    # simplest t_hat
    #t_hat = 2*np.sqrt(np.log(p))
    
    #return np.abs(W) >= t_hat

    # BH FDR correction
    upper_triangle = vectorize_matrix(W)
    p_vals = np.array([G_func(np.abs(t)) for t in upper_triangle])
    
    p_vals = np.triu(symmetrize_from_vector(p_vals, p))
    
    return p_vals <= alpha

def apply_cai_algorithm_windowed(args, data):
    #global_p_vals = []
    global_test_vals = []
    for i in tqdm(range(0, data.shape[0]-2*args.window_size, args.step_size)):
        data_window = data[i:i+2*args.window_size, :]
        if args.fix_pre:
            first_data_window = data[0:args.window_size, :]
            second_data_window = data[i+args.window_size:i+2*args.window_size, :]
            data_window = np.concatenate((first_data_window, second_data_window), axis=0)
        #beta_hats_x, beta_hats_y = perform_regression(data)
        residuals_x, residuals_y, beta_hats_x, beta_hats_y  = perform_regression(data_window)
        middle = data_window.shape[0]//2
        #residuals_x = calculate_residuals(beta_hats_x, data_window[:middle, :])
        residuals_x_cov_corrected = bias_corrected_residual_covariance(residuals_x, beta_hats_x)
        T_x = calculate_T(residuals_x_cov_corrected)
        theta_x = calculate_theta(residuals_x_cov_corrected, beta_hats_x, N=data_window[:middle, :].shape[0])

        residuals_y = calculate_residuals(beta_hats_y, data_window[middle:, :])
        residuals_y_cov_corrected = bias_corrected_residual_covariance(residuals_y, beta_hats_y)
        T_y = calculate_T(residuals_y_cov_corrected)
        theta_y = calculate_theta(residuals_y_cov_corrected, beta_hats_y, N=data_window[middle:, :].shape[0])

        W = calculate_standardized_stat(T_x, T_y, theta_x, theta_y)
        M = calculate_global_stat(W)
        threshold = indicator_global_stat(M, p=data.shape[1], alpha=0.01)

        # print(M)
        # print(threshold)
        # print(M>threshold)

        local_test = np.triu(indicator_local_stat(W, p=data.shape[1], alpha=0.01).astype(int), k=1)

        global_test_vals.append(M)
        #global_p_vals.append()
    
    return np.array(global_test_vals), local_test

@jit(nopython=True)
def get_mean(data):
    data_mean = np.zeros(data.shape[1])

    for col in range(data.shape[1]):
        mu = np.mean(data[:, col])
        data_mean[col] = mu
    
    return data_mean

@jit(nopython=True)
def get_theta_variance(data, C):
    dim = C.shape[0]
    theta = np.zeros((dim, dim))
    mu = get_mean(data)
    for i in range(dim):
        for j in range(dim):
            summand = 0
            for k in range(data.shape[0]):
                first = (data[k, i] - mu[i])
                second = (data[k,j] - mu[j])
                third = C[i,j]
                summand += np.power((first*second-third), 2)
            summand /= data.shape[0]
            theta[i,j] = summand
    
    return theta

@jit(nopython=True)
def get_T_variance(theta_one, theta_two, C_one, C_two, n1, n2):
    dim = C_one.shape[0]
    T = np.zeros((dim, dim))

    for i in range(dim):
        for j in range(i, dim):
            top = np.power((C_one[i,j] - C_two[i,j]), 2)
            bottom = (theta_one[i,j]*(1/n1)) + (theta_two[i,j]*(1/n2))
            T[i,j] = top/bottom
    
    return T


def apply_cai_variance_windowed(args, data):
    global_test_vals = []

    for i in tqdm(range(0, data.shape[0]-2*args.window_size, args.step_size)):
        data_window = data[i:i+2*args.window_size, :]
        middle = data_window.shape[0]//2
        X_data = data_window[:middle, :]
        Y_data = data_window[middle:, :]
        X_cov = np.cov(X_data.T, bias=True)
        Y_cov = np.cov(Y_data.T, bias=True)
        X_theta = get_theta_variance(X_data, X_cov)
        Y_theta = get_theta_variance(Y_data, Y_cov)
        T = get_T_variance(X_theta, Y_theta, X_cov, Y_cov, n1=X_data.shape[0], n2=Y_data.shape[0])
        M = np.max(T)
        global_test_vals.append(M)
        threshold = indicator_global_stat(M, p=data.shape[1], alpha=0.05)

    
    return np.array(global_test_vals), T

    
def perform_simulation_batch(args):
    # run a batch of 50 simulations/results with a specified simulation model
    # only local test
    # should do step_size = 1 because it's easier
    # save everything to files - I guess
    print("\n*******************************************************************************")
    print("Performing Batch Simulation of {} with Dim = {}, Window = {}".format(args.sim_type, args.dim, args.window_size))
    seeds_list = np.arange(50, 60)
    sim_results_path = os.path.join(args.results_path, "simulation_results_cai")
    if not os.path.isdir(sim_results_path):
        os.mkdir(sim_results_path)
    sim_type_path = os.path.join(sim_results_path, args.sim_type+"_"+str(args.dim))
    print(sim_type_path)
    if not os.path.isdir(sim_type_path):
        os.mkdir(sim_type_path)
    for seed in seeds_list:
        np.random.seed(seed)
        save_path = os.path.join(sim_type_path, str(seed))
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        data_full = resolve_data(args, save_path=save_path)
        print(data_full.shape)
        global_test_vals, _ = apply_cai_algorithm_windowed(args, data_full)
        print()
        np.savetxt(os.path.join(save_path, "global_test_vals.csv"), global_test_vals, delimiter=',')
    print("*******************************************************************************")
    print("Done!")

def perform_simulation_batch_variance(args):
    # run a batch of 50 simulations/results with a specified simulation model
    # only local test
    # should do step_size = 1 because it's easier
    # save everything to files - I guess
    print("\n*******************************************************************************")
    print("Performing Batch Simulation of {} with Dim = {}, Window = {}".format(args.sim_type, args.dim, args.window_size))
    seeds_list = np.arange(50, 70)
    sim_results_path = os.path.join(args.results_path, "simulation_results_cai")
    if not os.path.isdir(sim_results_path):
        os.mkdir(sim_results_path)
    sim_type_path = os.path.join(sim_results_path, args.sim_type+"_"+str(args.dim))
    print(sim_type_path)
    if not os.path.isdir(sim_type_path):
        os.mkdir(sim_type_path)
    for seed in seeds_list:
        np.random.seed(seed)
        save_path = os.path.join(sim_type_path, str(seed))
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        data_full = resolve_data(args, save_path=save_path)
        print("DATA SHAPE {}".format(data_full.shape))
        global_test_vals, local_test_vals = apply_cai_variance_windowed(args, data_full)
        np.savetxt(os.path.join(save_path, "global_test_vals.csv"), global_test_vals, delimiter=',')
    print("*******************************************************************************")
    print("Done!")

def perform_single_run(args):
    data_full = resolve_data(args, save_path=None)
    print(data_full.shape)
    global_test_vals, _ = apply_cai_algorithm_windowed(args, data_full)
    plt.plot(global_test_vals)
    plt.show()

def precision_recall_sims(args):
    print("***************************")
    print("Performing Prec/Recall Simulations")
    args.N = 101
    args.window_size = 100
    args.step_size = 2
    args.sim = 1
    simulation_prefixes = ['cai_model_one', 'cai_model_three', 
                        'orthogonal', 'cholesky', 'sparse_cholesky', 'cai_model_four']
    no_change_prefixes = ['cai_model_one_no_change', 'cai_model_three_no_change', 
                        'orthogonal_no_change', 'cholesky_no_change', 'sparse_cholesky_no_change', 'cai_model_four_no_change']
    simulation_dims = ['_6', '_10', '_15', '_20', '_30', '_50', '_100']
    M_s_non_orthog = [2, 3, 4, 6, 8, 10, 20]
    M_s_orthog = [2, 2, 3, 4, 5, 10, 20]
    sim_results_path = os.path.join(args.results_path, "simulation_results_precision_cai")
    seeds = np.arange(50, 100)
    if not os.path.isdir(sim_results_path):
        os.mkdir(sim_results_path)
    for sim_type in simulation_prefixes:
        args.sim_type = sim_type
        for k, dim in enumerate(simulation_dims):
            if 'orthogonal' in args.sim_type:
                args.M = M_s_orthog[k]
            else:
                args.M = M_s_non_orthog[k]
            args.dim = int(dim[1:])
            sim_type_path = os.path.join(sim_results_path, args.sim_type+str(dim))
            if not os.path.isdir(sim_type_path):
                os.mkdir(sim_type_path)
            print(sim_type_path)
            for seed in seeds:
                np.random.seed(seed)
                save_path = os.path.join(sim_type_path, str(seed))
                if not os.path.isdir(save_path):
                    os.mkdir(save_path)
                data_full = resolve_data(args, save_path=save_path)
                print(data_full.shape)
                global_test_vals, T = apply_cai_variance_windowed(args, data_full)
                print()
                np.savetxt(os.path.join(save_path, "global_test_vals.csv"), global_test_vals, delimiter=',')
                np.savetxt(os.path.join(save_path, "local_test_vals.csv"), T, delimiter=',')

    for sim_type in no_change_prefixes:
        args.sim_type = sim_type
        for k, dim in enumerate(simulation_dims):
            if 'orthogonal' in args.sim_type:
                args.M = M_s_orthog[k]
            else:
                args.M = M_s_non_orthog[k]
            args.dim = int(dim[1:])
            sim_type_path = os.path.join(sim_results_path, args.sim_type+str(dim))
            if not os.path.isdir(sim_type_path):
                os.mkdir(sim_type_path)
            print(sim_type_path)
            for seed in seeds:
                np.random.seed(seed)
                save_path = os.path.join(sim_type_path, str(seed))
                if not os.path.isdir(save_path):
                    os.mkdir(save_path)
                data_full = resolve_data(args, save_path=save_path)
                print(data_full.shape)
                global_test_vals, T = apply_cai_variance_windowed(args, data_full)
                print()
                np.savetxt(os.path.join(save_path, "global_test_vals.csv"), global_test_vals, delimiter=',')
                np.savetxt(os.path.join(save_path, "local_test_vals.csv"), T, delimiter=',')
    
    print("***************************")
    print("Done!")

def precision_recall_sims_precision(args):
    print("***************************")
    print("Performing Prec/Recall Simulations")
    args.N = 101
    args.window_size = 100
    args.step_size = 2
    args.sim = 1
    simulation_prefixes = ['cai_model_one', 'cai_model_three', 
                        'orthogonal', 'cholesky']
    no_change_prefixes = ['cai_model_one_no_change', 'cai_model_three_no_change', 
                        'orthogonal_no_change', 'cholesky_no_change']
    simulation_dims = ['_6', '_10', '_15', '_20', '_30', '_50', '_100']
    M_s_non_orthog = [2, 3, 4, 6, 8, 10, 20]
    M_s_orthog = [2, 2, 3, 4, 5, 10, 20]
    sim_results_path = os.path.join(args.results_path, "simulation_results_precision_cai")
    seeds = np.arange(50, 100)
    if not os.path.isdir(sim_results_path):
        os.mkdir(sim_results_path)
    for sim_type in simulation_prefixes:
        args.sim_type = sim_type
        for k, dim in enumerate(simulation_dims):
            if 'orthogonal' in args.sim_type:
                args.M = M_s_orthog[k]
            else:
                args.M = M_s_non_orthog[k]
            args.dim = int(dim[1:])
            sim_type_path = os.path.join(sim_results_path, args.sim_type+str(dim))
            if not os.path.isdir(sim_type_path):
                os.mkdir(sim_type_path)
            print(sim_type_path)
            for seed in seeds:
                np.random.seed(seed)
                save_path = os.path.join(sim_type_path, str(seed))
                if not os.path.isdir(save_path):
                    os.mkdir(save_path)
                data_full = resolve_data(args, save_path=save_path)
                print(data_full.shape)
                global_test_vals, T = apply_cai_algorithm_windowed(args, data_full)
                print()
                np.savetxt(os.path.join(save_path, "global_test_vals.csv"), global_test_vals, delimiter=',')
                #np.savetxt(os.path.join(save_path, "local_test_vals.csv"), T, delimiter=',')

    for sim_type in no_change_prefixes:
        args.sim_type = sim_type
        for k, dim in enumerate(simulation_dims):
            if 'orthogonal' in args.sim_type:
                args.M = M_s_orthog[k]
            else:
                args.M = M_s_non_orthog[k]
            args.dim = int(dim[1:])
            sim_type_path = os.path.join(sim_results_path, args.sim_type+str(dim))
            if not os.path.isdir(sim_type_path):
                os.mkdir(sim_type_path)
            print(sim_type_path)
            for seed in seeds:
                np.random.seed(seed)
                save_path = os.path.join(sim_type_path, str(seed))
                if not os.path.isdir(save_path):
                    os.mkdir(save_path)
                data_full = resolve_data(args, save_path=save_path)
                print(data_full.shape)
                global_test_vals, T = apply_cai_algorithm_windowed(args, data_full)
                print()
                np.savetxt(os.path.join(save_path, "global_test_vals.csv"), global_test_vals, delimiter=',')
                #np.savetxt(os.path.join(save_path, "local_test_vals.csv"), T, delimiter=',')
    
    print("***************************")
    print("Done!")


def precision_recall_sims_precision_runtimes(args):
    print("***************************")
    print("Performing Runtime Simulations")
    args.N = 101
    args.window_size = 100
    args.step_size = 2
    args.sim = 1
    # simulation_prefixes = ['cai_model_one', 'cai_model_three', 
    #                     'orthogonal', 'cholesky']
    # no_change_prefixes = ['cai_model_one_no_change', 'cai_model_three_no_change', 
    #                     'orthogonal_no_change', 'cholesky_no_change']
    simulation_prefixes = ['cai_model_four']
    no_change_prefixes = []
    simulation_dims = ['_6', '_10', '_15', '_20', '_30', '_50', '_100']
    #simulation_dims = ['_100']
    M_s_non_orthog = [2, 3, 4, 6, 8, 10, 20]
    M_s_orthog = [2, 2, 3, 4, 5, 10, 20]
    sim_results_path = os.path.join(args.results_path, "simulation_results_precision_cai")
    seeds = np.arange(50, 60)
    if not os.path.isdir(sim_results_path):
        os.mkdir(sim_results_path)
    runtimes = []
    for sim_type in simulation_prefixes:
        args.sim_type = sim_type
        for k, dim in enumerate(simulation_dims):
            if 'orthogonal' in args.sim_type:
                args.M = M_s_orthog[k]
            else:
                args.M = M_s_non_orthog[k]
            args.dim = int(dim[1:])
            sim_type_path = os.path.join(sim_results_path, args.sim_type+str(dim))
            if not os.path.isdir(sim_type_path):
                os.mkdir(sim_type_path)
            print(sim_type_path)
            curr_runtimes = []
            for seed in seeds:
                np.random.seed(seed)
                save_path = os.path.join(sim_type_path, str(seed))
                if not os.path.isdir(save_path):
                    os.mkdir(save_path)
                data_full = resolve_data(args, save_path=save_path)
                print(data_full.shape)
                start_time = time.time()
                global_test_vals, T = apply_cai_algorithm_windowed(args, data_full)
                end_time = time.time()
                curr_runtimes.append(end_time-start_time)
                print()
                #np.savetxt(os.path.join(save_path, "global_test_vals.csv"), global_test_vals, delimiter=',')
                #np.savetxt(os.path.join(save_path, "local_test_vals.csv"), T, delimiter=',')
            runtimes.append(np.mean(curr_runtimes))

    for sim_type in no_change_prefixes:
        args.sim_type = sim_type
        for k, dim in enumerate(simulation_dims):
            if 'orthogonal' in args.sim_type:
                args.M = M_s_orthog[k]
            else:
                args.M = M_s_non_orthog[k]
            args.dim = int(dim[1:])
            sim_type_path = os.path.join(sim_results_path, args.sim_type+str(dim))
            if not os.path.isdir(sim_type_path):
                os.mkdir(sim_type_path)
            print(sim_type_path)
            for seed in seeds:
                np.random.seed(seed)
                save_path = os.path.join(sim_type_path, str(seed))
                if not os.path.isdir(save_path):
                    os.mkdir(save_path)
                data_full = resolve_data(args, save_path=save_path)
                print(data_full.shape)
                global_test_vals, T = apply_cai_algorithm_windowed(args, data_full)
                print()
                #np.savetxt(os.path.join(save_path, "global_test_vals.csv"), global_test_vals, delimiter=',')
                #np.savetxt(os.path.join(save_path, "local_test_vals.csv"), T, delimiter=',')
    np.savetxt(os.path.join(args.results_path, 'runtimes/runtimes_cai.csv'), np.array(runtimes), delimiter=',')
    print("***************************")
    print("Done!")


def main():
    args = get_args()
    np.random.seed(args.random_seed)
    data = resolve_data(args)
    beta_hats_x, beta_hats_y = perform_regression(data)
    middle = data.shape[0]//2
    residuals_x = calculate_residuals(beta_hats_x, data[:middle, :])
    residuals_x_cov_corrected = bias_corrected_residual_covariance(residuals_x, beta_hats_x)
    T_x = calculate_T(residuals_x_cov_corrected)
    theta_x = calculate_theta(residuals_x_cov_corrected, beta_hats_x, N=data[:middle, :].shape[0])

    residuals_y = calculate_residuals(beta_hats_y, data[middle:, :])
    residuals_y_cov_corrected = bias_corrected_residual_covariance(residuals_y, beta_hats_y)
    T_y = calculate_T(residuals_y_cov_corrected)
    theta_y = calculate_theta(residuals_y_cov_corrected, beta_hats_y, N=data[middle:, :].shape[0])

    W = calculate_standardized_stat(T_x, T_y, theta_x, theta_y)
    M = calculate_global_stat(W)
    threshold = indicator_global_stat(M, p=data.shape[1], alpha=0.01)

    print(M)
    print(threshold)
    print(M>threshold)

    local_test = np.triu(indicator_local_stat(W, p=data.shape[1], alpha=0.01).astype(int), k=1)
    print(local_test)



if __name__ == '__main__':
    #main()
    args = get_args()
    #data = resolve_data(args)
    #apply_cai_variance_windowed(args, data)
    perform_simulation_batch(args)
    #perform_simulation_batch_variance(args)
    #precision_recall_sims(args)
    #precision_recall_sims_precision(args)
    #precision_recall_sims_precision_runtimes(args)
    #perform_single_run(args)