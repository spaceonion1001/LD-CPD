import argparse
import os
import numpy as np
np.random.seed(24)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from precision_cpd import PrecisionCPD
from simulate import sim_changepoint_mv_normal_cholesky, sim_changepoint_mv_normal_no_decomp, sim_changepoint_mv_normal_orthogonal, sim_changepoint_mv_normal_ldlt
from utils import load_alaska_data, scale_data, load_hjandrews_data, create_fig_dir

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lam', type=float, default=1e-1)
    parser.add_argument('--sim', type=int, default=1)
    parser.add_argument('--M', type=int, default=2)
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--step_size', type=int, default=50)
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--data', type=str, default='alaska')
    parser.add_argument('--data_path', type=str, default='/home/dink/Documents/Research/data')
    parser.add_argument('--fig_path', type=str, default='/home/dink/Documents/Research/Correlation-Changepoint-Detection/figs')
    parser.add_argument('--basic_test', type=int, default=0)
    parser.add_argument('--full_basis', type=int, default=1)
    parser.add_argument('--local', type=int, default=1)
    parser.add_argument('--sim_type', type=str, default='orthogonal')
    parser.add_argument('--sim_scale', type=float, default=1.5)
    parser.add_argument('--include_l1', type=int, default=1)
    parser.add_argument('--split_variance', type=int, default=0)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--beta', type=float, default=5e-3)
    parser.add_argument('--t', type=float, default=2.0)
    args = parser.parse_args()

    return args

def resolve_data(args):
    if bool(args.sim):
        if args.sim_type == 'orthogonal':
            return sim_changepoint_mv_normal_orthogonal(sim_scale=args.sim_scale, M=args.M, dim=args.dim, N=args.N)[1].T
        elif args.sim_type == 'cholesky':
            return scale_data(sim_changepoint_mv_normal_cholesky(dim=args.dim, N=args.N, num_coeffs_change=1, scale=args.sim_scale))
        elif args.sim_type == 'ldlt':
            return scale_data(sim_changepoint_mv_normal_ldlt(dim=args.dim, N=args.N, num_coeffs_change=1, scale=args.sim_scale))
        else:
            return sim_changepoint_mv_normal_no_decomp(dim=args.dim, N=args.N, num_coeffs_change=1, scale=args.sim_scale).T
    else:
        if args.data == 'alaska':
            return load_alaska_data(args)
        elif args.data == 'hjandrews':
            return load_hjandrews_data(args)
        else:
            print("Error: Dataset not understood")
            exit(0)

if __name__ == '__main__':
    args = get_args()
    data_full = resolve_data(args)
    print(data_full.shape)
    fig_dir_path = create_fig_dir(args.fig_path)
    model = PrecisionCPD(args)
    if bool(args.basic_test):
        lrt_vals, p_vals = model.perform_lrt_covariance(data_full.T)
        print(lrt_vals.shape)
        print(np.argmax(lrt_vals))
        plt.plot(lrt_vals)
        plt.xlabel('Time')
        plt.ylabel('Test Statistic')
        plt.savefig(os.path.join(fig_dir_path, 'lrt_cov_basic_win{}_step{}.png'.format(args.window_size, args.step_size)))
        
    else:
        data_train = data_full[0:int(0.66*len(data_full)), :]
        model.fit_glasso(data_train)
        model.construct_basis_matrices()
        if bool(args.local):
            print("Local Test")
            lrt_vals_all, p_vals_all = model.perform_lrt_local(data_full.T)
            for i in range(lrt_vals_all.shape[1]):
                plt.plot(lrt_vals_all[:, i])
                plt.xlabel('Time')
                plt.ylabel('Test Statistic {}'.format(i))
                plt.savefig(os.path.join(fig_dir_path, 'lrt_local_i{}_M{}_win{}_step{}_lam{}_full{}_sim{}.png'.format(i, args.M, args.window_size, 
                                                                                                                    args.step_size, args.lam, 
                                                                                                                    args.full_basis, args.sim)))
                plt.close()

        else:
            print("Global Test")
            lrt_vals, p_vals = model.perform_lrt_global(data_full.T)
            plt.plot(lrt_vals)
            plt.xlabel('Time')
            plt.ylabel('Test Statistic')
            plt.savefig(os.path.join(fig_dir_path, 'lrt_global_M{}_win{}_step{}_lam{}_full{}_sim{}.png'.format(args.M, args.window_size, 
                                                                                                                args.step_size, args.lam, 
                                                                                                                args.full_basis, args.sim)))
            plt.close()
        model.print_clusters_rv()
    