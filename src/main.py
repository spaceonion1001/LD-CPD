import argparse
import os
from venv import create
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
sns.set()

from precision_cpd import PrecisionCPD
from simulate import *
from utils import difference_data, load_alaska_data, scale_data, load_hjandrews_data, create_fig_dir, load_holiday_farm_data, load_tohoku_data, load_stock_market_data

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lam', type=float, default=3e-1)
    parser.add_argument('--sim', type=int, default=1)
    parser.add_argument('--M', type=int, default=2)
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--dim', type=int, default=16)
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--data', type=str, default='alaska')
    parser.add_argument('--data_path', type=str, default='/home/dink/Documents/Research/data')
    parser.add_argument('--fig_path', type=str, default='/home/dink/Documents/Research/Correlation-Changepoint-Detection/figs')
    parser.add_argument('--results_path', type=str, default='/home/dink/Documents/Research/Correlation-Changepoint-Detection/results')
    parser.add_argument('--results_filename', type=str, default="lrt_results.csv")
    parser.add_argument('--basic_test', type=int, default=0)
    parser.add_argument('--full_basis', type=int, default=1)
    parser.add_argument('--local', type=int, default=1)
    parser.add_argument('--sim_type', type=str, default='cholesky')
    parser.add_argument('--sim_scale', type=float, default=1.5)
    parser.add_argument('--include_l1', type=int, default=1)
    parser.add_argument('--split_variance', type=int, default=0)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--beta', type=float, default=5e-3)
    parser.add_argument('--t', type=float, default=1.0)
    parser.add_argument('--save_test_stat', type=int, default=0)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--train_percent', type=float, default=0.25)
    parser.add_argument('--single_test', type=int, default=1)
    parser.add_argument('--num_coeffs_change', type=int, default=1)
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

def perform_single_test(args):
    np.random.seed(args.random_seed)
    fig_dir_path = create_fig_dir(args.fig_path)
    results_dir_path = create_fig_dir(args.results_path)
    data_full = resolve_data(args, results_dir_path)
    print(data_full.shape)
    model = PrecisionCPD(args)
    # simple covariance test using sample covariance
    if bool(args.basic_test):
        lrt_vals, p_vals = model.perform_lrt_covariance(data_full.T)
        print(lrt_vals.shape)
        print(np.argmax(lrt_vals))
        plt.plot(lrt_vals)
        plt.xlabel('Time')
        plt.ylabel('Test Statistic')
        plt.savefig(os.path.join(fig_dir_path, 'lrt_cov_basic_win{}_step{}.png'.format(args.window_size, args.step_size)))
    # our algorithm            
    else:
        data_train = data_full[0:int(args.train_percent*len(data_full)), :]
        model.fit_glasso(data_train)
        model.construct_basis_matrices()
        # local test
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
            if bool(args.save_test_stat):
                test_results = np.hstack([lrt_vals_all, p_vals_all])
                np.savetxt(os.path.join(results_dir_path, args.results_filename), test_results, delimiter=',')
        # global test
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
            if bool(args.save_test_stat):
                test_results = np.vstack([lrt_vals, p_vals]).T
                np.savetxt(os.path.join(results_dir_path, 'lrt_global.csv'), test_results, delimiter=',')
        model.print_clusters_rv()

def perform_simulation_batch(args):
    # run a batch of 50 simulations/results with a specified simulation model
    # only local test
    # should do step_size = 1 because it's easier
    # save everything to files - I guess
    print("\n*******************************************************************************")
    print("Performing Batch Simulation of {} with Dim = {}, M = {}, Scale = {}, Window = {}, Lam = {}".format(args.sim_type, args.dim, args.M, args.sim_scale, args.window_size, args.lam))
    seeds_list = np.arange(50, 75)
    sim_results_path = os.path.join(args.results_path, "simulation_results")
    if not os.path.isdir(sim_results_path):
        os.mkdir(sim_results_path)
    sim_type_path = os.path.join(sim_results_path, args.sim_type)
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
        model = PrecisionCPD(args)
        data_train = data_full[0:int(args.train_percent*len(data_full)), :]
        model.fit_glasso(data_train)
        model.construct_basis_matrices()
        lrt_vals_all, p_vals_all = model.perform_lrt_local(data_full.T)
        model.save_matrices_simulations(save_path)
        print()
        np.savetxt(os.path.join(save_path, "lrt_vals.csv"), lrt_vals_all, delimiter=',')
        np.savetxt(os.path.join(save_path, "p_vals.csv"), p_vals_all, delimiter=',')
        model.print_clusters_rv()
    print("*******************************************************************************")
    print("Done!")

if __name__ == '__main__':
    args = get_args()
    if bool(args.single_test):
        perform_single_test(args)
    else:
        perform_simulation_batch(args)

    
    


# python src/main.py --sim 0 --data alaska --M 2 --window_size 50 --step_size 1 --lam 1e-3 --train_percent 0.5 --full_basis 1 --split_variance 1
# python src/main.py --sim 1 --sim_type cholesky --dim 50 --sim_scale 1.5 --window_size 100 --step_size 1 --lam 3e-1 --single_test 0 --N 500 --M 8 --t 1.0 --train_percent 0.2 --full_basis 1 --beta 6e-3
# python src/main.py --sim 1 --sim_type orthogonal_mult_coeff --dim 50 --sim_scale 1.5 --window_size 100 --step_size 1 --lam 3e-1 --single_test 0 --N 500 --M 5 --t 1.0 --train_percent 0.2 --full_basis 0 --beta 6e-3 --num_coeffs_change 2
# python src/main.py --sim 1 --sim_type cai_model_three --dim 50 --window_size 100 --step_size 1 --lam 3e-1 --single_test 0 --N 500 --M 8 --t 1.0 --train_percent 0.2 --full_basis 1 --beta 6e-3

# python src/main.py --sim 1 --sim_type ldlt --dim 50 --window_size 100 --step_size 1 --lam 1e-1 --single_test 0 --N 500 --M 10 --t 1.0 --train_percent 0.4 --full_basis 1 --beta 6e-3 --num_coeffs_change 2
# python src/main.py --sim 1 --sim_type orthogonal_mult_coeff --dim 50 --window_size 100 --step_size 1 --lam 1e-1 --single_test 0 --N 500 --M 10 --t 1.0 --train_percent 0.4 --full_basis 1 --beta 6e-3 --num_coeffs_change 2
# python src/main.py --sim 1 --sim_type cai_model_three --dim 50 --window_size 100 --step_size 1 --lam 1e-1 --single_test 0 --N 500 --M 10 --t 1.0 --train_percent 0.4 --full_basis 1 --beta 6e-3
# python src/main.py --sim 1 --sim_type cholesky --dim 50 --window_size 100 --step_size 1 --lam 3e-1 --single_test 0 --N 500 --M 10 --t 1.0 --train_percent 0.45 --full_basis 1 --beta 6e-3 --num_coeffs_change 2