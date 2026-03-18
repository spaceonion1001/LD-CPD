import numpy as np
from numpy.linalg import inv as inv
from simulate import sim_changepoint_mv_normal_cholesky, sim_changepoint_mv_normal_ldlt, sim_changepoint_var_process, changepoint_cai_model_one, changepoint_cai_model_three, anderson_sim_with_residual
from utils import scale_data, difference_data, vectorize_matrix, symmetrize_from_vector, load_mesonet_data
from tqdm import tqdm
from scipy.special import comb, polygamma, erf, erfinv
import seaborn as sns

from simulate import *
from utils import difference_data, load_alaska_data, scale_data, load_hjandrews_data, create_fig_dir, load_holiday_farm_data, load_tohoku_data, load_stock_market_data, load_sap_data, load_mesonet_pressure_data
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

matplotlib.use('Agg')

from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import logging
import csv
rpy2_logger.setLevel(logging.ERROR)

r = robjects.r
rpy2.robjects.numpy2ri.activate()
utils = importr('utils')
utils.chooseCRANmirror(ind=1)
# R package names
packnames = ('scalreg', 'fastclime')

# R vector of strings
from rpy2.robjects.vectors import StrVector

# Selectively install what needs to be installed.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))
scalreg = importr('scalreg')
fastclime = importr('fastclime')

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lam', type=float, default=5e-2)
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--N', type=int, default=500)
    parser.add_argument('--M', type=int, default=2)
    parser.add_argument('--sim', type=int, default=1)
    parser.add_argument('--sim_type', type=str, default='cai_model_three')
    parser.add_argument('--data', type=str, default='alaska')
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--data_fname', type=str, default='mesonet_test_out.csv')
    parser.add_argument('--sim_scale', type=float, default=0.8)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--post_window_size', type=int, default=20)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--burn_in', type=int, default=100)
    parser.add_argument('--buffer_size', type=int, default=50)
    parser.add_argument('--pi_0', type=float, default=0.05)
    parser.add_argument('--num_coeffs_change', type=int, default=2)
    parser.add_argument('--results_path', type=str, default='./results')
    parser.add_argument('--percent', type=float, default=0.25, help='Percent to use for scaling data')
    parser.add_argument('--kesh_d', type=int, default=10)
    parser.add_argument('--resid_type', type=str, choices=['unstructured', 'block'], default='unstructured', help='Residual Type')
    parser.add_argument('--num_indices', type=int, default=4)
    parser.add_argument('--train_percent', type=float, default=0.1)
    parser.add_argument('--single_test', type=int, default=0)
    parser.add_argument('--results_fldr_name', type=str, default=None)
    parser.add_argument('--results_filename', type=str, default=None)
    parser.add_argument('--sap', type=int, default=0)
    parser.add_argument('--alt', type=int, default=0)
    parser.add_argument('--estimator', type=str, choices=['clime', 'scalreg'], default='clime')
    parser.add_argument('--auto_lambda', type=int, default=0)
    parser.add_argument('--load_lambdas', type=str, default=None)
    args = parser.parse_args()

    return args

def resolve_data(args, save_path=None, data_seed=42):
    if bool(args.sim):
        if args.sim_type == 'orthogonal_small':
            return sim_changepoint_mv_normal_orthogonal(sim_scale=args.sim_scale, M=args.M, dim=args.dim, N=args.N, save_path=save_path, data_seed=data_seed)[1].T
        elif args.sim_type == 'orthogonal_mult_coeff':
            return sim_changepoint_mv_normal_orthogonal_mult_coeff(sim_scale=args.sim_scale, num_coeffs_change=args.num_coeffs_change, M=args.M, dim=args.dim, N=args.N, save_path=save_path)[1].T
        elif args.sim_type == 'orthogonal_cross_block':
            return sim_changepoint_mv_normal_orthogonal_cross_block(sim_scale=args.sim_scale, M=args.M, dim=args.dim, N=args.N, save_path=save_path, data_seed=data_seed)[1].T
        elif args.sim_type == 'orthogonal_multiple_block':
            return sim_changepoint_mv_normal_orthogonal_multiple_block(sim_scale=args.sim_scale, M=args.M, dim=args.dim, N=args.N, save_path=save_path, data_seed=data_seed)[1].T
        elif args.sim_type == 'orthogonal_hard':
            return sim_changepoint_mv_normal_orthogonal_hard(sim_scale=args.sim_scale, M=args.M, dim=args.dim, N=args.N, save_path=save_path, data_seed=data_seed)[1].T
        elif args.sim_type == 'orthogonal_cross_hard':
            return sim_changepoint_mv_normal_orthogonal_cross_hard(sim_scale=args.sim_scale, M=args.M, dim=args.dim, N=args.N, save_path=save_path, data_seed=data_seed)[1].T
        elif args.sim_type == 'cholesky':
            return scale_data(sim_changepoint_mv_normal_cholesky(dim=args.dim, N=args.N, num_coeffs_change=args.num_coeffs_change, scale=args.sim_scale, save_path=save_path), percent=args.percent)
        elif args.sim_type == 'ldlt':
            return scale_data(sim_changepoint_mv_normal_ldlt(dim=args.dim, N=args.N, num_coeffs_change=args.num_coeffs_change, scale=args.sim_scale, save_path=save_path), percent=args.percent)
        elif args.sim_type == 'var_process':
            return scale_data(difference_data(sim_changepoint_var_process(dim=args.dim, N=args.N, num_coeffs_change=args.num_coeffs_change, scale=args.sim_scale, save_path=save_path)), percent=args.percent)
        elif args.sim_type == 'cai_model_one':
            return scale_data(changepoint_cai_model_one(args, dim=args.dim, N=args.N, save_path=save_path), end_idx=args.window_size)
        elif args.sim_type == 'cai_model_one_extra':
            return scale_data(changepoint_cai_model_one(args, dim=args.dim, N=args.N, save_path=save_path), end_idx=args.window_size)
        elif args.sim_type == 'cai_model_three':
            return scale_data(changepoint_cai_model_three(args, dim=args.dim, N=args.N, save_path=save_path), end_idx=args.window_size)
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
            return load_stock_market_data(args)
        elif args.data == 'mesonet':
            return scale_data(load_mesonet_data(args), percent=None, end_idx=args.window_size)
        elif args.data == 'mesonet_pressure':
            return scale_data(load_mesonet_pressure_data(args), percent=None, end_idx=args.window_size)
        elif args.data == 'sap':
            return scale_data(load_sap_data(args), percent=None, end_idx=args.window_size)
        else:
            print("Error: Dataset not understood")
            exit(0)

def load_lambda_override(args, seed=None):
    if not args.load_lambdas:
        return None
    candidates = []
    if seed is not None:
        candidates.append(os.path.join(args.load_lambdas, str(seed), "chosen_lambda.csv"))
        sim_dir = args.sim_type + "_" + str(args.dim)
        if args.sim_type == 'anderson_residual':
            sim_dir = args.sim_type + "_" + args.resid_type + "_" + str(args.dim)
        candidates.append(os.path.join(args.load_lambdas, sim_dir, str(seed), "chosen_lambda.csv"))
    else:
        candidates.append(os.path.join(args.load_lambdas, "chosen_lambda.csv"))

    csv_path = None
    for cand in candidates:
        if os.path.isfile(cand):
            csv_path = cand
            break
    if csv_path is None:
        print("Warning: load_lambdas did not find chosen_lambda.csv; using args.lam.")
        return None
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if len(rows) < 2:
            print("Warning: chosen_lambda.csv missing data row; using args.lam.")
            return None
        chosen = rows[-1][-1]
        return float(chosen)
    except Exception:
        print("Warning: failed to parse chosen_lambda.csv; using args.lam.")
        return None


class KeshOnline:

    def __init__(self, args, data, seed=None):
        self.args = args
        self.data_full = data
        self.N = args.burn_in
        self.w = args.window_size
        self.post_window_size = args.post_window_size
        self.buffer = args.buffer_size
        self.pi_0 = args.pi_0
        self.p = self.data_full.shape[1]
        self.lam = args.lam
        self.seed = seed
        if args.estimator == 'clime' and args.load_lambdas and not bool(args.auto_lambda):
            loaded_lam = load_lambda_override(args, seed=seed)
            if loaded_lam is not None:
                self.lam = loaded_lam
                print("Loaded lambda override {}".format(self.lam))
        elif args.estimator == 'clime' and bool(args.auto_lambda):
            print("Auto lambda enabled: ignoring args.lam and any loaded lambda.")

        self.D_hat = []

        self.data = self.data_full[self.N:, :]

        if args.estimator == 'clime':
            self.clime_init = self.fastclime_init_fn(args, self.data_full[0:self.N, :])
        else:
            self.clime_init = self.scalreg_init_fn(args, self.data_full[0:self.N, :])
        self.g1 = calc_g1(self.w)
        #print("G1 {}".format(self.g1))
        self.g2 = calc_g2(self.w)
        #print("G2 {}".format(self.g2))
        self.rhat0 = calc_rhat(self.clime_init, self.p)
        self.T0 = calc_T_t(X=self.data_full, omega_hat=self.clime_init, r_hat=self.rhat0, w=self.w, p=self.p, t=0, g1=self.g1, g2=self.g2)
        self.critical_value = inv_Q_func(self.pi_0)

        if self.args.alt:
            print("*** ALT VERSION ***")
            self.test_stats = self.iterate_alt(self.data)
        else:
            self.test_stats = self.iterate(self.data)

    def scalreg_init_fn(self, args, data_minimal):
        nrow, ncol = data_minimal.shape
        X = r.matrix(data_minimal, nrow=nrow, ncol=ncol)
        reg_soln = scalreg.scalreg(X, lam0="univ")
        reg_soln_dict = dict(zip(reg_soln.names, list(reg_soln)))
        clime_est = reg_soln_dict['precision']
        return clime_est

    def fastclime_init_fn(self, args, data_minimal):
        nrow, ncol = data_minimal.shape
        X = r.matrix(data_minimal, nrow=nrow, ncol=ncol)

        if args.auto_lambda:
            clime_out = fastclime.fastclime(X)
        else:
            clime_out = fastclime.fastclime(X, self.lam, 100)

        clime_soln = dict(zip(clime_out.names, list(clime_out)))
        lambdamtx = np.array(clime_soln['lambdamtx'])
        icovlist = clime_soln['icovlist']

        if args.auto_lambda:
            # sample size and dimension
            n = nrow
            p = ncol

            # sample covariance
            S = np.cov(data_minimal, rowvar=False, bias=True)
            row_lambdas = np.median(lambdamtx, axis=1)

            def bic_score(theta):
                theta = (theta + theta.T) / 2
                sign, logdet = np.linalg.slogdet(theta)
                if sign <= 0 or not np.isfinite(logdet):
                    return np.inf

                nz = np.sum(np.abs(theta) > 1e-8)
                offdiag_nz = max(nz - p, 0)
                k = p + (offdiag_nz / 2.0)

                negloglik = n * (np.trace(S @ theta) - logdet)
                return negloglik + k * np.log(n)

            # build τ grid from paper
            tau_grid = [
                (10 ** (-1 + j / 10.0)) * np.sqrt(np.log(p) / n)
                for j in range(20)
            ]

            best_tau = None
            best_idx = None
            best_score = np.inf

            for tau in tau_grid:
                try:
                    idx = int(np.argmin(np.abs(row_lambdas - tau)))
                    theta = np.array(icovlist[idx])
                except Exception:
                    continue

                score = bic_score(theta)

                if score < best_score:
                    best_score = score
                    best_tau = tau
                    best_idx = idx

            if best_tau is None or best_idx is None:
                print("Warning: BIC grid search failed; falling back to args.lam.")
            else:
                self.lam = float(row_lambdas[best_idx])
                print("Auto-selected lambda via BIC grid (tau {}, path {})".format(best_tau, self.lam))

        # Ensure lambda is reachable for all columns (avoid fastclime warning)
        # Use the max of per-column minima across the path.
        min_required = float(np.max(np.min(lambdamtx, axis=0)))
        if self.lam < min_required:
            print("Clamping lambda from {} to {} to satisfy column limits".format(self.lam, min_required))
            self.lam = min_required

        max_attempts = 5
        bump_factor = 1.5
        clime_est = None
        last_err = None
        for attempt in range(max_attempts):
            try:
                select_out = fastclime.fastclime_selector(lambdamtx, icovlist, self.lam)
                select_soln = dict(zip(select_out.names, list(select_out)))
                clime_est = np.array(select_soln['icov'])
                break
            except Exception as exc:
                last_err = exc
                self.lam = self.lam * bump_factor
                print("Warning: fastclime selector failed; bumping lambda to {}".format(self.lam))

        if clime_est is None:
            if args.auto_lambda and best_idx is not None:
                clime_est = np.array(icovlist[best_idx])
            else:
                raise last_err

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

        test_stats = []
        t = 0
        for t in tqdm(range(0, data.shape[0]-self.post_window_size, self.args.step_size)):
            T_t = calc_T_t(X=data, omega_hat=curr_clime, r_hat=curr_rhat, w=self.post_window_size, p=self.p, t=t, g1=self.g1, g2=self.g2)
            test_stats.append(T_t)

            indicator_fn = (T_t >= self.critical_value)
        
        return np.array(test_stats)
    
    def iterate_alt(self, data):
        curr_clime = self.clime_init

        test_stats = []
        t = 0
        for t in tqdm(range(0, data.shape[0]-self.post_window_size, self.args.step_size)):
            E_hat = calc_E_hat(data, omega_hat=curr_clime, t=t, p=self.p, w=self.post_window_size)
            test_stats.append(np.linalg.norm(E_hat, ord=np.inf))


        return np.array(test_stats)
        

#@jit(nopython=True)
def calc_E_hat(data, omega_hat, t, p, w):
    summand_mat = np.zeros((p, p))
    for r in range(1, w):
        X_t_r = np.expand_dims(data[t+r, :], 1)
        dot_prod = (np.dot(omega_hat, X_t_r)@np.dot(omega_hat, X_t_r).T)
        subtr = omega_hat
        first = (dot_prod-subtr)/np.sqrt(w)
        summand_mat += first
    second = np.zeros((p, p))
    for u in range(p):
        for v in range(p):
            val = omega_hat[u, u]*omega_hat[v,v]+(omega_hat[u,v]**2)
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

@jit(nopython=True)
def calc_l4_norm(mat):
    sum = 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            sum += np.abs(mat[i, j])**4
    
    return sum ** (1./4)

def calc_y_s_t_w(X, omega_hat, w, s, t):
    div = 1/(w*omega_hat[s, s])
    summand = 0
    for r in range(1, w):
        val = np.inner(X[t+r, :], omega_hat[:, s])**2
        summand += val
    
    return div*summand

def calc_T_t(X, omega_hat, r_hat, w, p, t, g1, g2):
    top = 0
    w_s = []
    for s in range(p):
        curr_y = calc_y_s_t_w(X, omega_hat, w=w, s=s, t=t)
        curr_f = calc_f(curr_y)
        top += (curr_f - g1)
        w_s.append(w*curr_y)

    bot = g2 * calc_hw(r_hat)

    return top/bot

def inv_Q_func(pi_0):
    q_func = np.sqrt(2)*erfinv(1-2*pi_0)
    return q_func


def perform_single_run(args):
    data_full = resolve_data(args, save_path=None)
    print(data_full.shape)
    save_root = args.results_path+"_kesh"
    if args.alt:
        save_root = args.results_path+"_kesh_alt"
    if args.estimator == 'clime':
        save_root = save_root + "_clime"
    save_path = os.path.join(save_root, args.data)
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)
    model = KeshOnline(args, data_full, seed=args.random_seed)
    global_test_vals = model.test_stats
    np.savetxt(os.path.join(save_path, args.data_fname+"_global_test_vals.csv"), global_test_vals, delimiter=',')
    plt.plot(global_test_vals)
    plt.savefig(os.path.join(save_path, args.data_fname+".png"))
    plt.close()

    

def perform_simulation_batch(args):
    print("\n*******************************************************************************")
    print("Performing Batch Simulation of {} with Dim = {}, Window = {}".format(args.sim_type, args.dim, args.window_size))
    seeds_list = np.arange(51, 70)
    sim_results_root = os.path.join(args.results_path, "simulation_results_kesh")
    if args.alt:
        sim_results_root = os.path.join(args.results_path, "simulation_results_kesh_alt")
    if args.estimator == 'clime':
        sim_results_root = sim_results_root + "_clime"
    if not os.path.isdir(sim_results_root):
        os.makedirs(sim_results_root, exist_ok=True)
    sim_type_path = os.path.join(sim_results_root, args.sim_type+"_"+str(args.dim))
    if args.sim_type == 'anderson_residual':
        sim_type_path = os.path.join(sim_results_root, args.sim_type+"_"+args.resid_type+"_"+str(args.dim))
    print(sim_type_path)
    if not os.path.isdir(sim_type_path):
        os.makedirs(sim_type_path, exist_ok=True)
    for seed in seeds_list:
        np.random.seed(seed)
        save_path = os.path.join(sim_type_path, str(seed))
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)
        data_full = resolve_data(args, save_path=save_path, data_seed=seed)
        print(data_full.shape)
        model = KeshOnline(args, data_full, seed=seed)
        global_test_vals = model.test_stats
        print("Test Vals Shape", global_test_vals.shape)
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
    sim_results_root = os.path.join(args.results_path, "sap_results_kesh")
    if args.alt:
        sim_results_root = os.path.join(args.results_path, "sap_results_kesh_alt")
    if args.estimator == 'clime':
        sim_results_root = sim_results_root + "_clime"
    if not os.path.isdir(sim_results_root):
        os.makedirs(sim_results_root, exist_ok=True)
    sim_type_path = os.path.join(sim_results_root, "sap_500"+"_"+str(args.dim))
    print(sim_type_path)
    if not os.path.isdir(sim_type_path):
        os.makedirs(sim_type_path, exist_ok=True)
    for seed in seeds_list:
        np.random.seed(seed)
        save_path = os.path.join(sim_type_path, str(seed))
        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)
        data_full = resolve_data(args, save_path=save_path, data_seed=seed)
        # if 'orthogonal' in args.sim_type:
        #     H_s, data_full = data_full
        #     data_full = data_full.T
        print(data_full.shape)
        dim_list = np.arange(0, data_full.shape[1])
        chosen_idxs = np.random.choice(dim_list, size=args.dim, replace=False)
        data_full = data_full[:, chosen_idxs]
        model = KeshOnline(args, data_full, seed=seed)
        global_test_vals = model.test_stats
        print("Test Vals Shape", global_test_vals.shape)
        print()
        np.savetxt(os.path.join(save_path, "global_test_vals.csv"), global_test_vals, delimiter=',')
        np.savetxt(os.path.join(save_path, "chosen_idxs.csv"), chosen_idxs, delimiter=',')
        plt.plot(global_test_vals)
        plt.savefig(os.path.join(save_path, "test_stat.png"))
        plt.close()

    print("*******************************************************************************")
    print("Done!")

def main():
    #np.random.seed(24)
    args = get_args()
    print("Window Size {} Post Window Size {} Lamba {} Train Percent {}".format(args.window_size, args.post_window_size, args.lam, args.train_percent))
    if args.single_test:
        perform_single_run(args)
    elif args.sap:
        perform_sap_batch(args)
    else:
        perform_simulation_batch(args)



if __name__ == '__main__':
    main()

    
