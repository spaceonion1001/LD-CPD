from pickletools import read_string1
import numpy as np
from numpy.linalg import inv as inv
from scipy.optimize import least_squares, minimize
from simulate import sim_changepoint_mv_normal_cholesky
from utils import scale_data

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lam', type=float, default=1e-1)
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--sim_scale', type=float, default=0.8)

    args = parser.parse_args()

    return args


def resolve_data(args):
    return scale_data(sim_changepoint_mv_normal_cholesky(dim=args.dim, N=args.N, num_coeffs_change=1, scale=args.sim_scale))


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
    
    return np.array(beta_hats_x), np.array(beta_hats_y)

def calculate_residuals(beta_hat, data):
    """
    TODO
    """

    pass


def main():
    np.random.seed(42)
    args = get_args()
    data = resolve_data(args)
    beta_hats_x, beta_hats_y = perform_regression(data)
    print(beta_hats_x)
    print(beta_hats_y)
    




if __name__ == '__main__':
    main()