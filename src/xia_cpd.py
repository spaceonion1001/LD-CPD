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

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lam', type=float, default=1e-1)
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--M', type=int, default=2)
    parser.add_argument('--sim', type=int, default=1)
    parser.add_argument('--sim_type', type=str, default='cai_model_three')
    parser.add_argument('--data', type=str, default='alaska')
    parser.add_argument('--data_path', type=str, default='/home/dink/Documents/Research/data')
    parser.add_argument('--sim_scale', type=float, default=0.8)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--results_path', type=str, default='/home/dink/Documents/Research/Correlation-Changepoint-Detection/results')
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
        else:
            return sim_changepoint_mv_normal_no_decomp(dim=args.dim, N=args.N, num_coeffs_change=1, scale=args.sim_scale, save_path=save_path).T
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
            return load_stock_market_data(args)
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
    for i in range(dim):
        X_m_i = np.delete(X_data, i, axis=1)
        X_i = X_data[:, i]
        sig_i_i_x = np.var(X_i)
        x_lam = kd*np.power(sig_i_i_x*x_log_dim, 0.5)
        D_i_x = np.power(np.diag(np.cov(X_m_i.T, bias=False)), -0.5)
        res_x_i = minimize(loss_func, x_0, method='BFGS', args=(X_data, i, x_lam)).x
        beta_hat_x_i = D_i_x*res_x_i

        Y_m_i = np.delete(Y_data, i, axis=1)
        Y_i = Y_data[:, i]
        sig_i_i_y = np.var(Y_i)
        y_lam = kd*np.power(sig_i_i_y*y_log_dim, 0.5)
        D_i_y = np.power(np.diag(np.cov(Y_m_i.T, bias=False)), -0.5)
        res_y_i = minimize(loss_func, x_0, method='BFGS', args=(Y_data, i, y_lam)).x
        beta_hat_y_i = D_i_y*res_y_i

        beta_hats_x.append(beta_hat_x_i)
        beta_hats_y.append(beta_hat_y_i)
    
    # return should be (dim, dim-1) shape
    # since we need one regression for each dim, and is vector size of dim-1
    return np.array(beta_hats_x), np.array(beta_hats_y)

def calculate_residuals(beta_hats, data):
    # this is for one half of the data
    N = data.shape[0]
    dim = data.shape[1]
    residuals = np.zeros((N, dim))
    for k in range(N):
        for i in range(dim):
            X_m_i = np.delete(data, i, axis=1)
            X_m_i_k = X_m_i[k, :]
            X_i_k = data[k, i]
            X_m_i_k_min_mu = X_m_i_k - np.mean(X_m_i.T, 1)
            X_i_k_min_mu = X_i_k - np.mean(data[:, i])
            pred = X_m_i_k_min_mu.dot(beta_hats[i])
            epsilon = X_i_k_min_mu - pred
            residuals[k, i] = epsilon
    
    return residuals

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

def calculate_T(residuals_cov_corrected):
    dim = residuals_cov_corrected.shape[0]
    T = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            top = residuals_cov_corrected[i, j]
            bottom = residuals_cov_corrected[i,i]*residuals_cov_corrected[j,j]
            T[i, j] = top/bottom
    
    return T

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

def calculate_standardized_stat(T_x, T_y, theta_x, theta_y):
    W = (T_x - T_y)/np.sqrt(theta_x+theta_y)

    return W

def calculate_global_stat(W):
    M = np.max(W**2)

    return M

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
        beta_hats_x, beta_hats_y = perform_regression(data)
        middle = data_window.shape[0]//2
        residuals_x = calculate_residuals(beta_hats_x, data_window[:middle, :])
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
    
    return np.array(global_test_vals)

    
def perform_simulation_batch(args):
    # run a batch of 50 simulations/results with a specified simulation model
    # only local test
    # should do step_size = 1 because it's easier
    # save everything to files - I guess
    print("\n*******************************************************************************")
    print("Performing Batch Simulation of {} with Dim = {}, Window = {}".format(args.sim_type, args.dim, args.window_size))
    seeds_list = np.arange(50, 75)
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
        global_test_vals = apply_cai_algorithm_windowed(args, data_full)
        print()
        np.savetxt(os.path.join(save_path, "global_test_vals.csv"), global_test_vals, delimiter=',')
    print("*******************************************************************************")
    print("Done!")

def perform_single_run(args):
    data_full = resolve_data(args, save_path=None)
    print(data_full.shape)
    global_test_vals = apply_cai_algorithm_windowed(args, data_full)
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
    #main()
    args = get_args()
    perform_simulation_batch(args)
    #perform_single_run(args)