import numpy as np
from numpy.linalg import inv as inv
from simulate import sim_changepoint_mv_normal_cholesky, sim_changepoint_mv_normal_ldlt, sim_changepoint_var_process, changepoint_cai_model_one, changepoint_cai_model_three, anderson_sim_with_residual
from utils import scale_data, difference_data, vectorize_matrix, symmetrize_from_vector
from tqdm import tqdm
from scipy.stats import norm
from simulate import *
from utils import difference_data, load_alaska_data, scale_data, load_hjandrews_data, create_fig_dir, load_holiday_farm_data, load_tohoku_data, load_stock_market_data, load_mesonet_data, load_sap_data, load_mesonet_pressure_data
from numba import jit
import time
from datetime import timedelta
import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages

import matplotlib.pyplot as plt
import matplotlib


from sklearn.linear_model import Lasso

matplotlib.use("Agg")

r = robjects.r
rpy2.robjects.numpy2ri.activate()
utils = importr('utils')
utils.chooseCRANmirror(ind=1)

# R vector of strings
from rpy2.robjects.vectors import StrVector

scalreg = importr('scalreg')

import argparse
import warnings
warnings.filterwarnings('ignore')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lam', type=float, default=5e-2)
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--N', type=int, default=500)
    parser.add_argument('--M', type=int, default=2)
    parser.add_argument('--sim', type=int, default=1)
    parser.add_argument('--sim_type', type=str, default='cai_model_one')
    parser.add_argument('--data', type=str, default='alaska')
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--data_fname', type=str, default='mesonet_test_out.csv')
    parser.add_argument('--sim_scale', type=float, default=0.2)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--train_percent', type=float, default=0.1)
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--post_window_size', type=int, default=20)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--num_coeffs_change', type=int, default=2)
    parser.add_argument('--results_path', type=str, default='./results')
    parser.add_argument('--fix_pre', type=int, default=1)
    parser.add_argument('--resid_type', type=str, choices=['unstructured', 'block'], default='unstructured', help='Residual Type')
    parser.add_argument('--num_indices', type=int, default=4)
    parser.add_argument('--single_test', type=int, default=0)
    parser.add_argument('--results_fldr_name', type=str, default=None)
    parser.add_argument('--results_filename', type=str, default=None)
    parser.add_argument('--sap', type=int, default=0)
    args = parser.parse_args()

    return args


def resolve_data(args, save_path=None, data_seed=42):
    if bool(args.sim):
        if args.sim_type == 'orthogonal_small':
            return sim_changepoint_mv_normal_orthogonal(sim_scale=args.sim_scale, M=args.M, dim=args.dim, N=args.N, save_path=save_path, data_seed=data_seed)[1].T
        elif args.sim_type == 'orthogonal_cross_block':
            return sim_changepoint_mv_normal_orthogonal_cross_block(sim_scale=args.sim_scale, M=args.M, dim=args.dim, N=args.N, save_path=save_path, data_seed=data_seed)[1].T
        elif args.sim_type == 'orthogonal_multiple_block':
            return sim_changepoint_mv_normal_orthogonal_multiple_block(sim_scale=args.sim_scale, M=args.M, dim=args.dim, N=args.N, save_path=save_path, data_seed=data_seed)[1].T
        elif args.sim_type == 'orthogonal_hard':
            return sim_changepoint_mv_normal_orthogonal_hard(sim_scale=args.sim_scale, M=args.M, dim=args.dim, N=args.N, save_path=save_path, data_seed=data_seed)[1].T
        elif args.sim_type == 'orthogonal_cross_hard':
            return sim_changepoint_mv_normal_orthogonal_cross_hard(sim_scale=args.sim_scale, M=args.M, dim=args.dim, N=args.N, save_path=save_path, data_seed=data_seed)[1].T
        elif args.sim_type == 'orthogonal_mult_coeff':
            return sim_changepoint_mv_normal_orthogonal_mult_coeff(sim_scale=args.sim_scale, num_coeffs_change=args.num_coeffs_change, M=args.M, dim=args.dim, N=args.N, save_path=save_path)[1].T
        elif args.sim_type == 'cholesky':
            return scale_data(sim_changepoint_mv_normal_cholesky(dim=args.dim, N=args.N, num_coeffs_change=args.num_coeffs_change, scale=args.sim_scale, save_path=save_path))
        elif args.sim_type == 'ldlt':
            return scale_data(sim_changepoint_mv_normal_ldlt(dim=args.dim, N=args.N, num_coeffs_change=args.num_coeffs_change, scale=args.sim_scale, save_path=save_path))
        elif args.sim_type == 'var_process':
            return scale_data(difference_data(sim_changepoint_var_process(dim=args.dim, N=args.N, num_coeffs_change=args.num_coeffs_change, scale=args.sim_scale, save_path=save_path)))
        elif args.sim_type == 'cai_model_one':
            return scale_data(changepoint_cai_model_one(args, dim=args.dim, N=args.N, save_path=save_path), end_idx=args.window_size)
        elif args.sim_type == 'cai_model_three':
            return scale_data(changepoint_cai_model_three(args, dim=args.dim, N=args.N, save_path=save_path), end_idx=args.window_size)
        elif args.sim_type == 'cai_model_one_extra':
            return scale_data(changepoint_cai_model_one(args, dim=args.dim, N=args.N, save_path=save_path), end_idx=args.window_size)
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
            return scale_data(load_stock_market_data(args), args.train_percent)
        elif args.data == 'mesonet':
            return scale_data(load_mesonet_data(args), percent=None, end_idx=args.window_size)
        elif args.data == 'mesonet_pressure':
            return scale_data(load_mesonet_pressure_data(args), percent=None, end_idx=args.window_size)
        elif args.data == 'sap':
            return scale_data(load_sap_data(args), percent=None, end_idx=args.window_size)
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
        

def perform_regression(X_data, Y_data, kd=2):
    """
    Split data in half and perform the regression process on each column
    """
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
        D_i = np.diag(np.cov(X_m_i.T))
        x_predictors = (X_m_i-np.mean(X_m_i, axis=0))*np.sqrt(D_i)
        x_response = y_i-np.mean(y_i)
        x_gram = x_predictors.T.dot(x_predictors)
        x_lasso = Lasso(alpha=x_lam, fit_intercept=False, selection='random', precompute=x_gram).fit(x_predictors, x_response)
        residuals_x[:, i] = np.array(x_response) - x_lasso.predict(x_predictors)
        beta_hats_x_curr = np.sqrt(D_i)*np.array(x_lasso.coef_)

        beta_hats_x[i, :] = beta_hats_x_curr

        Y_m_i = np.delete(Y_data, i, axis=1)
        yy_i = Y_data[:, i]
        sig_i_i_y = np.var(yy_i)
        y_lam = kd*np.power(sig_i_i_y*y_log_dim, 0.5)
        D_i = np.diag(np.cov(Y_m_i.T))
        y_predictors = (Y_m_i-np.mean(Y_m_i, axis=0))*np.sqrt(D_i)
        y_response = yy_i-np.mean(yy_i)
        y_gram = y_predictors.T.dot(y_predictors)
        y_lasso = Lasso(alpha=y_lam, fit_intercept=False, selection='random', precompute=y_gram).fit(y_predictors, y_response)
        residuals_y[:, i] = np.array(y_response) - y_lasso.predict(y_predictors)
        beta_hats_y_curr = np.sqrt(D_i)*np.array(y_lasso.coef_)


        beta_hats_y[i, :] = beta_hats_y_curr

    return residuals_x, residuals_y, beta_hats_x, beta_hats_y

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
            idxs = []
            for j in range(dim):
                if j != i:
                    idxs.append(int(j))
            idxs = np.array(idxs)
            X_m_i = data[:, idxs]
            X_m_i_k = X_m_i[k, :]
            X_i_k = data[k, i]
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
            if bottom != 0.0:
                T[i, j] = top/bottom
            else:
                T[i, j] = top/1e-8 # numerical stability
    
    return T

@jit(nopython=True)
def calculate_theta(residuals_cov_corrected, beta_hats, N):
    dim = residuals_cov_corrected.shape[0]
    theta = np.zeros((dim, dim))

    for i in range(dim-1):
        for j in range(i+1, dim):
            top = (1 + (beta_hats.T[i, j]**2)*residuals_cov_corrected[i, i]/(residuals_cov_corrected[j,j] + 1e-12))
            bottom = N*residuals_cov_corrected[i,i]*residuals_cov_corrected[j,j]
            if bottom != 0.0:
                theta[i,j] = top/bottom
            else:
                theta[i,j] = top/1e-8 # numerical stability
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
    # BH FDR correction
    upper_triangle = vectorize_matrix(W)
    p_vals = np.array([G_func(np.abs(t)) for t in upper_triangle])
    
    p_vals = np.triu(symmetrize_from_vector(p_vals, p))
    
    return p_vals <= alpha

def apply_cai_algorithm_windowed(args, data):
    global_test_vals = []
    for i in tqdm(range(args.window_size, data.shape[0]-args.post_window_size, args.step_size)):
        if args.fix_pre:
            first_data_window = data[0:args.window_size, :] # fix the training window
            second_data_window = data[i:i+args.post_window_size, :] # move forward from i post_window_size ticks forward
        else:
            first_data_window = data[i:i+args.window_size, :]
            second_data_window = data[i+args.window_size:i+args.window_size+args.post_window_size, :]
        residuals_x, residuals_y, beta_hats_x, beta_hats_y  = perform_regression(X_data=first_data_window, Y_data=second_data_window)
        residuals_x_cov_corrected = bias_corrected_residual_covariance(residuals_x, beta_hats_x)
        T_x = calculate_T(residuals_x_cov_corrected)
        theta_x = calculate_theta(residuals_x_cov_corrected, beta_hats_x, N=first_data_window.shape[0])

        residuals_y_cov_corrected = bias_corrected_residual_covariance(residuals_y, beta_hats_y)
        T_y = calculate_T(residuals_y_cov_corrected)
        theta_y = calculate_theta(residuals_y_cov_corrected, beta_hats_y, N=second_data_window.shape[0])

        W = calculate_standardized_stat(T_x, T_y, theta_x, theta_y)
        M = calculate_global_stat(W)
        threshold = indicator_global_stat(M, p=data.shape[1], alpha=0.01)
        local_test = np.triu(indicator_local_stat(W, p=data.shape[1], alpha=0.01).astype(int), k=1)

        global_test_vals.append(M)
    
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
    print("\n*******************************************************************************")
    print("Performing Batch Simulation of {} with Dim = {}, Window = {}".format(args.sim_type, args.dim, args.window_size))
    seeds_list = np.arange(51, 70)
    sim_results_path = os.path.join(args.results_path, "simulation_results_cai")
    if not os.path.isdir(sim_results_path):
        os.mkdir(sim_results_path)
    sim_type_path = os.path.join(sim_results_path, args.sim_type+"_"+str(args.dim))
    if args.sim_type == 'anderson_residual':
        sim_type_path = os.path.join(sim_results_path, args.sim_type+"_"+args.resid_type+"_"+str(args.dim))
    print(sim_type_path)
    if not os.path.isdir(sim_type_path):
        os.mkdir(sim_type_path)
    for seed in seeds_list:
        print("Seed {}".format(seed))
        np.random.seed(seed)
        save_path = os.path.join(sim_type_path, str(seed))
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        data_full = resolve_data(args, save_path=save_path, data_seed=seed)
        print(data_full.shape)
        global_test_vals, _ = apply_cai_algorithm_windowed(args, data_full)
        print()
        np.savetxt(os.path.join(save_path, "global_test_vals.csv"), global_test_vals, delimiter=',')
        plt.plot(global_test_vals)
        plt.savefig(os.path.join(save_path, "test_stat.png"))
        plt.close()
    print("*******************************************************************************")
    print("Done!")

def perform_sap_batch(args):
    # run a batch of SAP500 data
    # batch constitutes a random sample of a subset of tickers
    print("\n*******************************************************************************")
    print("Performing SAP Run with Dim = {}, Window = {}, Post Window {}".format(args.dim, args.window_size, args.post_window_size))
    seeds_list = np.arange(50, 51)
    sim_results_path = os.path.join(args.results_path, "sap_results_cai")
    if not os.path.isdir(sim_results_path):
        os.mkdir(sim_results_path)
    sim_type_path = os.path.join(sim_results_path, "sap_500"+"_"+str(args.dim))
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
        dim_list = np.arange(0, data_full.shape[1])
        chosen_idxs = np.random.choice(dim_list, size=args.dim, replace=False)
        data_full = data_full[:, chosen_idxs]
        global_test_vals, _ = apply_cai_algorithm_windowed(args, data_full)
        print("Test Vals Shape", global_test_vals.shape)
        print()
        np.savetxt(os.path.join(save_path, "global_test_vals.csv"), global_test_vals, delimiter=',')
        np.savetxt(os.path.join(save_path, "chosen_idxs.csv"), chosen_idxs, delimiter=',')
        plt.plot(global_test_vals)
        plt.savefig(os.path.join(save_path, "test_stat.png"))
        plt.close()

    print("*******************************************************************************")
    print("Done!")

def perform_single_run(args):
    data_full = resolve_data(args, save_path=None)
    print(data_full.shape)
    global_test_vals, _ = apply_cai_algorithm_windowed(args, data_full)
    plt.plot(global_test_vals)
    plt.show()


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
    args = get_args()
    print("Window Size {} Post Window Size {} Train Percent {}".format(args.window_size, args.post_window_size, args.train_percent))
    if args.single_test:
        data_full = resolve_data(args)
        save_path = os.path.join(args.results_path+"_cai", args.data)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        print(data_full.shape)
        global_test_vals, _ = apply_cai_algorithm_windowed(args, data_full)
        print()
        np.savetxt(os.path.join(save_path, args.data_fname+"_global_test_vals.csv"), global_test_vals, delimiter=',')
        plt.plot(global_test_vals)
        plt.savefig(os.path.join(save_path, args.data_fname+"_global_test_vals.png"))
        plt.close()
    elif args.sap:
        perform_sap_batch(args)
    else:
        perform_simulation_batch(args)