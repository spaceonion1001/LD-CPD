from pickletools import read_string1
from cvxpy import vec
import numpy as np
from numpy.linalg import inv as inv
import scipy
from scipy.optimize import least_squares, minimize
from scipy.stats import chi2
from simulate import sim_changepoint_mv_normal_cholesky, sim_changepoint_mv_normal_ldlt, sim_changepoint_var_process, changepoint_cai_model_one, changepoint_cai_model_three, anderson_sim_with_residual
from utils import scale_data, difference_data, vectorize_matrix, symmetrize_from_vector, load_mesonet_data
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm
from scipy.stats import norm
from scipy.special import comb, polygamma, erf, erfinv
from sklearn.covariance import graphical_lasso, GraphicalLasso, GraphicalLassoCV
import seaborn as sns

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

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import logging
rpy2_logger.setLevel(logging.ERROR)

r = robjects.r
rpy2.robjects.numpy2ri.activate()
utils = importr('utils')
utils.chooseCRANmirror(ind=1)
# R package names
packnames = ('scalreg')

# R vector of strings
from rpy2.robjects.vectors import StrVector

# Selectively install what needs to be install.
# We are fancy, just because we can.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))
#clime = importr('clime')
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
    parser.add_argument('--data_fname', type=str, default='mesonet_test_out.csv')
    parser.add_argument('--sim_scale', type=float, default=0.8)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--burn_in', type=int, default=100)
    parser.add_argument('--buffer_size', type=int, default=50)
    parser.add_argument('--pi_0', type=float, default=0.05)
    parser.add_argument('--num_coeffs_change', type=int, default=2)
    parser.add_argument('--results_path', type=str, default='/home/dink/Documents/Research/Correlation-Changepoint-Detection/results')
    parser.add_argument('--percent', type=float, default=0.25, help='Percent to use for scaling data')
    parser.add_argument('--kesh_d', type=int, default=10)
    parser.add_argument('--resid_type', type=str, choices=['unstructured', 'block'], default='unstructured', help='Residual Type')
    parser.add_argument('--num_indices', type=int, default=4)
    parser.add_argument('--train_percent', type=float, default=0.25)
    parser.add_argument('--single_test', type=int, default=0)
    parser.add_argument('--results_fldr_name', type=str, default=None)
    parser.add_argument('--results_filename', type=str, default=None)
    args = parser.parse_args()

    return args

def resolve_data(args, save_path=None):
    if bool(args.sim):
        if args.sim_type == 'orthogonal_small':
            return sim_changepoint_mv_normal_orthogonal(sim_scale=args.sim_scale, M=args.M, dim=args.dim, N=args.N, save_path=save_path)[1].T
        elif args.sim_type == 'orthogonal_mult_coeff':
            return sim_changepoint_mv_normal_orthogonal_mult_coeff(sim_scale=args.sim_scale, num_coeffs_change=args.num_coeffs_change, M=args.M, dim=args.dim, N=args.N, save_path=save_path)[1].T
        elif args.sim_type == 'cholesky':
            return scale_data(sim_changepoint_mv_normal_cholesky(dim=args.dim, N=args.N, num_coeffs_change=args.num_coeffs_change, scale=args.sim_scale, save_path=save_path), percent=args.percent)
        elif args.sim_type == 'ldlt':
            return scale_data(sim_changepoint_mv_normal_ldlt(dim=args.dim, N=args.N, num_coeffs_change=args.num_coeffs_change, scale=args.sim_scale, save_path=save_path), percent=args.percent)
        elif args.sim_type == 'var_process':
            return scale_data(difference_data(sim_changepoint_var_process(dim=args.dim, N=args.N, num_coeffs_change=args.num_coeffs_change, scale=args.sim_scale, save_path=save_path)), percent=args.percent)
        elif args.sim_type == 'cai_model_one':
            return scale_data(changepoint_cai_model_one(args, dim=args.dim, N=args.N, save_path=save_path), percent=args.percent)
        elif args.sim_type == 'cai_model_three':
            return scale_data(changepoint_cai_model_three(args, dim=args.dim, N=args.N, save_path=save_path), percent=args.percent)
        elif args.sim_type == 'orthogonal_no_change':
            return sim_changepoint_mv_normal_orthogonal_no_change(sim_scale=args.sim_scale, M=args.M, dim=args.dim, N=args.N, save_path=save_path)[1].T
        elif args.sim_type == 'cholesky_no_change':
            return scale_data(sim_changepoint_mv_normal_cholesky_no_change(dim=args.dim, N=args.N, num_coeffs_change=args.num_coeffs_change, scale=args.sim_scale, save_path=save_path), percent=args.percent)
        elif args.sim_type == 'cai_model_one_no_change':
            data, prec_mat = changepoint_cai_model_one_no_change(dim=args.dim, N=args.N, save_path=save_path)
            return scale_data(data, percent=args.percent), prec_mat
            #return data, prec_mat
        elif args.sim_type == 'cai_model_three_no_change':
            return scale_data(changepoint_cai_model_three_no_change(dim=args.dim, N=args.N, save_path=save_path), percent=args.percent)
        elif args.sim_type == 'sparse_cholesky':
            return scale_data(sim_changepoint_mv_normal_cholesky(dim=args.dim, N=args.N, num_coeffs_change=args.num_coeffs_change, scale=args.sim_scale, save_path=save_path, sparse=True), percent=args.percent)
        elif args.sim_type == 'sparse_cholesky_no_change':
            return scale_data(sim_changepoint_mv_normal_cholesky_no_change(dim=args.dim, N=args.N, num_coeffs_change=args.num_coeffs_change, scale=args.sim_scale, save_path=save_path, sparse=True), percent=args.percent)
        elif args.sim_type == 'cai_model_four':
            return scale_data(changepoint_cai_model_four(dim=args.dim, N=args.N, save_path=save_path), percent=args.percent)
        elif args.sim_type == 'cai_model_four_no_change':
            return scale_data(changepoint_cai_model_four_no_change(dim=args.dim, N=args.N, save_path=save_path), percent=args.percent)
        elif args.sim_type == 'kesh':
            return changepoint_kesh_model(p=args.dim, d=args.kesh_d, N=args.N, beta=0.2, lambda_0=0.1)
        elif args.sim_type == 'anderson_residual':
            return scale_data(anderson_sim_with_residual(M=args.M, dim=args.dim, N=args.N, num_indices=args.num_indices, resid_type=args.resid_type, save_path=save_path), args.train_percent)
        else:
            print("Incorrect Simulation")
            exit(0)
    else:
        if args.data == 'alaska':
            return load_alaska_data(args)
        elif args.data == 'tohoku':
            return scale_data(load_tohoku_data(args), percent=args.percent)
        elif args.data == 'hjandrews':
            return load_hjandrews_data(args)
        elif args.data == 'holidayfarm':
            return load_holiday_farm_data(args)
        elif args.data == 'stocks':
            # python src/main.py --M 5 --lam 2e-5 --window_size 200 --step_size 1 --data stocks --local 1 --sim 0 --split_variance 0 --train_percent 0.6
            # stocks need a low lambda value I think
            return load_stock_market_data(args)
        elif args.data == 'mesonet':
            return scale_data(load_mesonet_data(args), args.train_percent)
        else:
            print("Error: Dataset not understood")
            exit(0)

class KeshOnline:

    def __init__(self, args, data):
        self.args = args
        self.data_full = data
        self.N = args.burn_in
        self.w = args.window_size
        self.buffer = args.buffer_size
        self.pi_0 = args.pi_0
        self.p = self.data_full.shape[1]
        self.lam = args.lam

        self.D_hat = []

        self.data = self.data_full[self.N:, :]

        self.clime_init = self.clime_init_fn(args, self.data_full[0:self.N, :])
        #self.prec_est = self.clime_init_fn(self.data_full)
        #self.glasso_est = GraphicalLasso(max_iter=100, alpha=self.lam, tol=1e-5, verbose=False).fit(self.data_full)
        #print(self.prec_mat, end='\n\n')
        #print(self.prec_est, end='\n\n')
        #print(self.glasso_est.precision_)
        self.glasso = GraphicalLasso(max_iter=100, alpha=self.lam, tol=1e-5, verbose=False).fit(self.data_full[0:self.N, :])
        #self.clime_init = self.glasso.precision_
        self.g1 = calc_g1(self.w)
        #print("G1 {}".format(self.g1))
        self.g2 = calc_g2(self.w)
        #print("G2 {}".format(self.g2))
        self.rhat0 = calc_rhat(self.clime_init, self.p)
        #self.rhat0 = calc_rhat(self.prec_mat, self.p)
        self.T0 = calc_T_t(X=self.data_full, omega_hat=self.clime_init, r_hat=self.rhat0, w=self.w, p=self.p, t=0, g1=self.g1, g2=self.g2)
        self.critical_value = inv_Q_func(self.pi_0)

        self.test_stats = self.iterate(self.data)
        # self.rhat0 = calc_rhat(self.glasso.precision_, self.p)
        # self.T0 = calc_T_t(X=self.data, omega_hat=self.glasso.precision_, r_hat=self.rhat0, w=self.w, p=self.p, t=0, g1=self.g1, g2=self.g2)
        # print(self.T0)
        # print(inv_Q_func(self.pi_0))
        #self.clime_init = self.clime_init_fn(self.data_full)

        # print(self.clime_init)
        # print(self.glasso.precision_)
        # print(args.prec_mat)
        # exit()

        

        #self.critical_value = self.critical_value_init(self.pi_0, self.p, self.w)

        #self.D_hat = 0
        #self.k = 0
        #self.t_hat_last = 0

        #print(scipy.linalg.norm(calc_E_hat(self.data, self.clime_init, t=0, p=self.p, w=self.w, prec_mat=self.prec_mat), ord=np.inf))
        #print(scipy.linalg.norm(calc_E_hat(self.data, self.glasso.precision_, t=0, p=self.p, w=self.w, prec_mat=self.prec_mat), ord=np.inf))
        #print(self.critical_value)

    def clime_init_fn(self, args, data_minimal):
        nrow, ncol = data_minimal.shape
        X = r.matrix(data_minimal, nrow=nrow, ncol=ncol)
        reg_soln = scalreg.scalreg(X, lam0=args.lam)
        reg_soln_dict = dict(zip(reg_soln.names, list(reg_soln)))
        clime_est = reg_soln_dict['precision']
        # nrow, ncol = data_minimal.shape
        # X = r.matrix(data_minimal, nrow=nrow, ncol=ncol)
        # clime_out = clime.fastclime(X, self.lam, 100)
        # clime_soln = dict(zip(clime_out.names, list(clime_out)))
        # lambdamtx = clime_soln['lambdamtx']
        # icovlist = clime_soln['icovlist']
        # select_out = clime.fastclime_selector(lambdamtx, icovlist, self.lam)
        # select_soln = dict(zip(select_out.names, list(select_out)))
        
        # clime_est = np.array(select_soln['icov'])
        return clime_est

    def critical_value_init(self, pi_0, p, w):
        """
        Eqn 3.1
        """

        inner_first = comb(p+1, 2, exact=True)
        first = 2*np.log(inner_first)
        second = np.log(np.log(inner_first))
        inner_log = np.log(1/(1-(pi_0/2)))
        third = 2*np.log(2*np.sqrt(np.pi)*inner_log)

        crit_val_squared = first-second-third

        return np.sqrt(crit_val_squared)
    
    def iterate(self, data):
        curr_clime = self.clime_init
        curr_rhat = self.rhat0
        curr_b = 0
        that_last = 0

        #for t in range(data.shape[0]):
        test_stats = []
        t = 0
        #while t+self.w <= data.shape[0]:
        for t in tqdm(range(0, data.shape[0]-self.w, self.args.step_size)):
            T_t = calc_T_t(X=data, omega_hat=curr_clime, r_hat=curr_rhat, w=self.w, p=self.p, t=t, g1=self.g1, g2=self.g2)
            test_stats.append(T_t)
            #T_t = calc_T_t(X=data, omega_hat=self.prec_mat, r_hat=self.rhat0, w=self.w, p=self.p, t=t, g1=self.g1, g2=self.g2)
            #print("Test Stat {} Critical Val {}".format(T_t, self.critical_value))
            indicator_fn = (T_t >= self.critical_value)
            #curr_b += 1
            # t += 1
            # if curr_b == self.buffer:
            #     curr_clime = self.clime_init_fn(self.args, data[that_last:t, :])
            #     #curr_clime = GraphicalLasso(max_iter=100, alpha=self.lam, tol=1e-5, verbose=False).fit(data[that_last:t, :]).precision_
            #     curr_rhat = calc_rhat(curr_clime, self.p)
            #     curr_b = 0
            # if not indicator_fn:
            #     curr_b += 1
            #     t += 1
            #     if curr_b == self.buffer:
            #         curr_clime = self.clime_init_fn(self.args, data[that_last:t, :])
            #         #curr_clime = GraphicalLasso(max_iter=100, alpha=self.lam, tol=1e-5, verbose=False).fit(data[that_last:t, :]).precision_
            #         curr_rhat = calc_rhat(curr_clime, self.p)
            #         curr_b = 0
            # else:
            #     that_last = t
            #     self.D_hat.append(t)
            #     curr_clime = self.clime_init_fn(self.args, data[t:t+self.N, :])
            #     #curr_clime = GraphicalLasso(max_iter=100, alpha=self.lam, tol=1e-5, verbose=False).fit(data[t:t+self.N, :]).precision_
            #     curr_rhat = calc_rhat(curr_clime, self.p)
            #     curr_b = 0
            #     t = t + self.N
        
        return np.array(test_stats)
        

#@jit(nopython=True)
def calc_E_hat(data, omega_hat, t, p, w):
    #prec_mat = inv(np.eye(p))
    summand_mat = np.zeros((p, p))
    for r in range(1, w):
        X_t_r = np.expand_dims(data[t+r, :], 1)
        dot_prod = (np.dot(omega_hat, X_t_r)@np.dot(omega_hat, X_t_r).T)
        #dot_prod = (np.dot(prec_mat, X_t_r)@np.dot(prec_mat, X_t_r).T)
        subtr = omega_hat
        #subtr = prec_mat
        first = (dot_prod-subtr)/np.sqrt(w)
        summand_mat += first
    second = np.zeros((p, p))
    for u in range(p):
        for v in range(p):
            val = omega_hat[u, u]*omega_hat[v,v]+(omega_hat[u,v]**2)
            #val = prec_mat[u, u]*prec_mat[v,v]+(prec_mat[u,v]**2)
            val = 1/np.sqrt(val)
            second[u,v] = val
    
    return summand_mat*second

def calc_g1(w):
    first = np.log(w/2)
    second = polygamma(0, w/2)

    return first - second

def calc_g2(w):
    first = polygamma(1, w/2)
    second = 2/w

    return np.sqrt(first - second)

def calc_rhat(omega_hat, p):
    r_hat = np.zeros(omega_hat.shape)

    for i in range(p):
        for j in range(p):
            top = omega_hat[i, j]
            bottom = np.sqrt(omega_hat[i,i]*omega_hat[j,j])
            r_hat[i,j] = top/bottom
    
    return r_hat

def calc_hw(r_hat):
    return calc_l4_norm(r_hat)**2

def calc_f(x):
    return x - 1 - np.log(x)

def calc_y_s_t_w(X, omega_hat, w, s, t):
    div = 1/(w*omega_hat[s, s])
    summand = 0
    for r in range(1, w):
        val = np.inner(X[t+r, :], omega_hat[:, s])**2
        summand += val
    
    return div*summand

@jit(nopython=True)
def calc_l4_norm(mat):
    sum = 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            sum += np.abs(mat[i, j])**4
    
    return sum ** (1./4)

def calc_T_t(X, omega_hat, r_hat, w, p, t, g1, g2):
    top = 0
    w_s = []
    for s in range(p):
        curr_y = calc_y_s_t_w(X, omega_hat, w=w, s=s, t=t)
        #print("Curr y {}".format(curr_y))
        curr_f = calc_f(curr_y)
        #print("Curr F {}".format(curr_f))
        top += (curr_f - g1)
        #print("Diff {}".format(curr_f-g1))
        #print("W*Y {} (this is Chi-square W dof)".format(w*curr_y))
        w_s.append(w*curr_y)
        #print("Top {}".format(top),end='\n\n')
    

    #plt.hist(w_s)
    # x = np.arange(0, 300, 0.01)
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # axes[0].plot(x, chi2.pdf(x, df=w))
    # axes[1].hist(w_s, density=True, bins=40)
    # axes[0].set_xlim(0.0, 300.0)
    # axes[1].set_xlim(0.0, 300.0)
    # axes[0].set_ylim(0.0, 0.06)
    # axes[1].set_ylim(0.0, 0.06)
    # plt.show()
    bot = g2 * calc_hw(r_hat)
    #print("Bot {}".format(bot))
    #print("Rhat {}".format(calc_hw(r_hat)))

    return top/bot

def inv_Q_func(pi_0):
    q_func = np.sqrt(2)*erfinv(1-2*pi_0)
    return q_func


def perform_single_run(args):
    data_full = resolve_data(args, save_path=None)
    print(data_full.shape)
    save_path = os.path.join(args.results_path+"_kesh", args.data)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    model = KeshOnline(args, data_full)
    global_test_vals = model.test_stats
    np.savetxt(os.path.join(save_path, args.data_fname+"_global_test_vals.csv"), global_test_vals, delimiter=',')
    plt.plot(global_test_vals)
    plt.savefig(os.path.join(save_path, args.data_fname+".png"))
    plt.close()

    

def perform_simulation_batch(args):
    # run a batch of 50 simulations/results with a specified simulation model
    # only local test
    # should do step_size = 1 because it's easier
    # save everything to files - I guess
    print("\n*******************************************************************************")
    print("Performing Batch Simulation of {} with Dim = {}, Window = {}".format(args.sim_type, args.dim, args.window_size))
    seeds_list = np.arange(50, 70)
    sim_results_path = os.path.join(args.results_path, "simulation_results_kesh")
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
        data_full = resolve_data(args, save_path=save_path)
        print(data_full.shape)
        model = KeshOnline(args, data_full)
        global_test_vals = model.test_stats
        print("Test Vals Shape", global_test_vals.shape)
        print()
        np.savetxt(os.path.join(save_path, "global_test_vals.csv"), global_test_vals, delimiter=',')
        plt.plot(global_test_vals)
        plt.savefig(os.path.join(save_path, "test_stat.png"))
        plt.close()
    print("*******************************************************************************")
    print("Done!")

def main():
    np.random.seed(24)
    args = get_args()
    if args.single_test:
        perform_single_run(args)
    else:
        perform_simulation_batch(args)



if __name__ == '__main__':
    main()

    