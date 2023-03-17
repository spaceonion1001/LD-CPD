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
import time
from datetime import timedelta


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
    parser.add_argument('--data_path', type=str, default='/Users/taylordinkins/Documents/Research/data/')
    parser.add_argument('--fig_path', type=str, default='./figs')
    parser.add_argument('--results_path', type=str, default='./results')
    parser.add_argument('--results_filename', type=str, default="lrt_results.csv")
    parser.add_argument('--basic_test', type=int, default=0)
    parser.add_argument('--full_basis', type=int, default=0)
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
    parser.add_argument('--prec_recall', type=int, default=0)
    parser.add_argument('--eps_matrices', type=int, default=0)
    parser.add_argument('--num_eps_mats', type=int, default=4)
    parser.add_argument('--runtimes', type=int, default=0)
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
            # return sim_changepoint_mv_normal_no_decomp(dim=args.dim, N=args.N, num_coeffs_change=1, scale=args.sim_scale, save_path=save_path).T
    else:
        if args.data == 'alaska':
            return load_alaska_data(args)
        elif args.data == 'tohoku':
            return scale_data(load_tohoku_data(args))
        elif args.data == 'hjandrews':
            return load_hjandrews_data(args)
        elif args.data == 'holidayfarm':
            # python src/main.py --sim 0 --data holidayfarm --eps_matrices 0 --lam 5e-2 --M 8 --full_basis 0 --single_test 1 --train_percent 0.2 --step_size 2 --num_eps_mats 2 --window_size 200 --save_test_stat 1
            return scale_data(load_holiday_farm_data(args))[22000:-10000, :]
        elif args.data == 'stocks':
            # python src/main.py --window_size 100 --sim 0 --data stocks --step 1 --window_size 100 --lam 5e-1 --full_basis 0 --M 6 --train_percent 0.4 --split_variance 0
            # python src/main.py --window_size 100 --sim 0 --data stocks --step 1 --window_size 100 --lam 5e-1 --full_basis 0 --M 10 --train_percent 0.4 --split_variance 0
            return scale_data(load_stock_market_data(args))
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
        #model.permute_blocks()

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
                model.save_matrices_simulations(results_dir_path)
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
    seeds_list = np.arange(50, 60)
    sim_results_path = os.path.join(args.results_path, "simulation_results")
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
        model = PrecisionCPD(args)
        data_train = data_full[0:int(args.train_percent*len(data_full)), :]
        model.fit_glasso(data_train)
        model.construct_basis_matrices()
        lrt_vals_all, _ = model.perform_lrt_local(data_full.T)
        model.save_matrices_simulations(save_path)
        print()
        np.savetxt(os.path.join(save_path, "lrt_vals.csv"), lrt_vals_all, delimiter=',')
        #np.savetxt(os.path.join(save_path, "p_vals.csv"), p_vals_all, delimiter=',')
        model.print_clusters_rv()
    print("*******************************************************************************")
    print("Done!")

def precision_recall_sims(args):
    print("***************************")
    print("Performing Prec/Recall Simulations")
    args.full_basis = 0
    args.split_variance = 0
    args.N = 101
    args.window_size = 100
    args.step_size = 2
    args.sim = 1
    args.lam = 1e-1
    args.train_percent = 0.4
    # simulation_prefixes = ['cai_model_one', 'cai_model_three', 
    #                     'orthogonal', 'cholesky']
    # no_change_prefixes = ['cai_model_one_no_change', 'cai_model_three_no_change', 
    #                     'orthogonal_no_change', 'cholesky_no_change']
    simulation_prefixes = ['cholesky']
    no_change_prefixes = ['cholesky_no_change']
    simulation_dims = ['_6', '_10', '_15', '_20', '_30', '_50', '_100']
    #M_s_non_orthog = [2, 3, 4, 6, 8, 10, 20]
    M_s_non_orthog = [3, 4, 5, 8, 10, 15, 25]
    M_s_orthog = [2, 2, 3, 4, 5, 10, 20]
    sim_results_path = os.path.join(args.results_path, "simulation_results_precision_cholesky")
    seeds = np.arange(50, 100)
    if not os.path.isdir(sim_results_path):
        os.mkdir(sim_results_path)
    for sim_type in simulation_prefixes:
        args.sim_type = sim_type
        for k, dim in enumerate(simulation_dims):
            if 'orthogonal' in args.sim_type:
                args.M = M_s_orthog[k]
                args.split_variance = 0
                args.full_basis = 0
            # elif 'cholesky' in args.sim_type:
            #     args.split_variance = 1
            #     args.full_basis = 1
            #     args.M = M_s_non_orthog[k]
            else:
                args.M = M_s_non_orthog[k]
            args.dim = int(dim[1:])
            if args.dim >= 50:
                args.iters = 200
                args.beta = 5e-3
            else:
                args.iters = 115
                args.beta = 8e-3
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
                model = PrecisionCPD(args)
                data_train = data_full[0:int(args.train_percent*len(data_full)), :]
                model.fit_glasso(data_train)
                model.construct_basis_matrices()
                lrt_vals_all, _ = model.perform_lrt_local(data_full.T)
                model.save_matrices_simulations(save_path)
                print()
                np.savetxt(os.path.join(save_path, "lrt_vals.csv"), lrt_vals_all, delimiter=',')
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
                model = PrecisionCPD(args)
                data_train = data_full[0:int(args.train_percent*len(data_full)), :]
                model.fit_glasso(data_train)
                model.construct_basis_matrices()
                lrt_vals_all, _ = model.perform_lrt_local(data_full.T)
                model.save_matrices_simulations(save_path)
                print()
                np.savetxt(os.path.join(save_path, "lrt_vals.csv"), lrt_vals_all, delimiter=',')
    
    print("***************************")
    print("Done!")

def precision_recall_sims_runtimes(args):
    print("***************************")
    print("Performing Runtime Calculation")
    args.full_basis = 0
    args.split_variance = 0
    args.N = 101
    args.window_size = 100
    args.step_size = 2
    args.sim = 1
    args.lam = 1e-1
    args.train_percent = 0.4
    # simulation_prefixes = ['cai_model_one', 'cai_model_three', 
    #                     'orthogonal', 'cholesky']
    # no_change_prefixes = ['cai_model_one_no_change', 'cai_model_three_no_change', 
    #                     'orthogonal_no_change', 'cholesky_no_change']
    simulation_prefixes = ['cai_model_four']
    #no_change_prefixes = ['cai_model_four_no_change']
    no_change_prefixes = []
    simulation_dims = ['_6', '_10', '_15', '_20', '_30', '_50', '_100']
    #M_s_non_orthog = [2, 3, 4, 6, 8, 10, 20]
    M_s_non_orthog = [3, 4, 5, 8, 10, 15, 25]
    M_s_orthog = [2, 2, 3, 4, 5, 10, 20]
    sim_results_path = os.path.join(args.results_path, "simulation_results_precision")
    seeds = np.arange(50, 60)
    if not os.path.isdir(sim_results_path):
        os.mkdir(sim_results_path)
    runtimes = []
    for sim_type in simulation_prefixes:
        args.sim_type = sim_type
        for k, dim in enumerate(simulation_dims):
            if 'orthogonal' in args.sim_type:
                args.M = M_s_orthog[k]
                args.split_variance = 0
                args.full_basis = 0
            # elif 'cholesky' in args.sim_type:
            #     args.split_variance = 1
            #     args.full_basis = 1
            #     args.M = M_s_non_orthog[k]
            else:
                args.M = M_s_non_orthog[k]
            args.dim = int(dim[1:])
            if args.dim >= 50:
                args.iters = 200
                args.beta = 5e-3
            else:
                args.iters = 115
                args.beta = 8e-3
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
                data_full = resolve_data(args, save_path=None)
                print(data_full.shape)
                start_time = time.time()
                model = PrecisionCPD(args)
                data_train = data_full[0:int(args.train_percent*len(data_full)), :]
                model.fit_glasso(data_train)
                model.construct_basis_matrices()
                lrt_vals_all, _ = model.perform_lrt_local(data_full.T)
                end_time = time.time()
                curr_runtimes.append(end_time-start_time)
                #model.save_matrices_simulations(save_path)
                print()
                #np.savetxt(os.path.join(save_path, "lrt_vals.csv"), lrt_vals_all, delimiter=',')
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
                start_time = time.monotonic()
                model = PrecisionCPD(args)
                data_train = data_full[0:int(args.train_percent*len(data_full)), :]
                model.fit_glasso(data_train)
                model.construct_basis_matrices()
                lrt_vals_all, _ = model.perform_lrt_local(data_full.T)
                #model.save_matrices_simulations(save_path)
                print()
                #np.savetxt(os.path.join(save_path, "lrt_vals.csv"), lrt_vals_all, delimiter=',')
    np.savetxt(os.path.join(args.results_path, 'runtimes/runtimes_ours.csv'), np.array(runtimes), delimiter=',')
    print("***************************")
    print("Done!")


if __name__ == '__main__':
    args = get_args()
    if bool(args.single_test):
        perform_single_test(args)
    elif bool(args.prec_recall):
        precision_recall_sims(args)
    elif bool(args.runtimes):
        precision_recall_sims_runtimes(args)
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

# python src/main.py --sim 0 --data stocks --lam 1e-2 --train_percent 0.4 --M 5 --beta 6e-3


# sims
# cai models
# dim 6
# python src/main.py --single_test 0 --num_coeffs_change 2 --train_percent 0.4 --window_size 100 --step_size 1 --sim 1 --lam 8e-2 --N 500 --M 2 --full_basis 0 --dim 6 --sim_type cai_model_three
# python src/main.py --single_test 0 --num_coeffs_change 2 --train_percent 0.4 --window_size 100 --step_size 1 --sim 1 --lam 8e-2 --N 500 --M 2 --full_basis 0 --dim 6 --sim_type cai_model_one
# dim 10
# python src/main.py --single_test 0 --num_coeffs_change 2 --train_percent 0.4 --window_size 100 --step_size 1 --sim 1 --lam 8e-2 --N 500 --M 3 --full_basis 0 --dim 10 --sim_type cai_model_three
# python src/main.py --single_test 0 --num_coeffs_change 2 --train_percent 0.4 --window_size 100 --step_size 1 --sim 1 --lam 8e-2 --N 500 --M 3 --full_basis 0 --dim 10 --sim_type cai_model_one
# dim 15
# python src/main.py --single_test 0 --num_coeffs_change 2 --train_percent 0.4 --window_size 100 --step_size 1 --sim 1 --lam 8e-2 --N 500 --M 4 --full_basis 0 --dim 15 --sim_type cai_model_three
# python src/main.py --single_test 0 --num_coeffs_change 2 --train_percent 0.4 --window_size 100 --step_size 1 --sim 1 --lam 8e-2 --N 500 --M 4 --full_basis 0 --dim 15 --sim_type cai_model_one
# dim 20
# python src/main.py --single_test 0 --num_coeffs_change 2 --train_percent 0.4 --window_size 100 --step_size 1 --sim 1 --lam 8e-2 --N 500 --M 6 --full_basis 0 --dim 20 --sim_type cai_model_three
# python src/main.py --single_test 0 --num_coeffs_change 2 --train_percent 0.4 --window_size 100 --step_size 1 --sim 1 --lam 8e-2 --N 500 --M 6 --full_basis 0 --dim 20 --sim_type cai_model_one
# dim 30
# python src/main.py --single_test 0 --num_coeffs_change 2 --train_percent 0.4 --window_size 100 --step_size 1 --sim 1 --lam 8e-2 --N 500 --M 8 --full_basis 0 --dim 30 --sim_type cai_model_three
# python src/main.py --single_test 0 --num_coeffs_change 2 --train_percent 0.4 --window_size 100 --step_size 1 --sim 1 --lam 8e-2 --N 500 --M 8 --full_basis 0 --dim 30 --sim_type cai_model_one
# dim 50
# python src/main.py --sim 1 --sim_type cai_model_three --dim 50 --window_size 100 --step_size 1 --lam 1e-1 --single_test 0 --N 500 --M 10 --t 1.0 --train_percent 0.4 --full_basis 1 --beta 6e-3
# python src/main.py --sim 1 --sim_type cai_model_one --dim 50 --window_size 100 --step_size 1 --lam 1e-1 --single_test 0 --N 500 --M 10 --t 1.0 --train_percent 0.4 --full_basis 1 --beta 6e-3
# dim 100
# python src/main.py --single_test 0 --num_coeffs_change 2 --train_percent 0.4 --window_size 100 --step_size 1 --sim 1 --lam 8e-2 --N 300 --M 20 --full_basis 0 --dim 100 --sim_type cai_model_three
# python src/main.py --single_test 0 --num_coeffs_change 2 --train_percent 0.4 --window_size 100 --step_size 1 --sim 1 --lam 8e-2 --N 300 --M 20 --full_basis 0 --dim 100 --sim_type cai_model_one


# non cai models
# dim 6
# python src/main.py --single_test 0 --num_coeffs_change 2 --train_percent 0.4 --window_size 100 --step_size 1 --sim 1 --lam 1e-1 --N 500 --M 2 --full_basis 0 --dim 6 --sim_type ldlt
# python src/main.py --single_test 0 --num_coeffs_change 2 --train_percent 0.4 --window_size 100 --step_size 1 --sim 1 --lam 1e-1 --N 500 --M 2 --full_basis 0 --dim 6 --sim_type cholesky
# dim 10
# python src/main.py --single_test 0 --num_coeffs_change 2 --train_percent 0.4 --window_size 100 --step_size 1 --sim 1 --lam 1e-1 --N 500 --M 3 --full_basis 0 --dim 10 --sim_type cholesky
# python src/main.py --single_test 0 --num_coeffs_change 2 --train_percent 0.4 --window_size 100 --step_size 1 --sim 1 --lam 1e-1 --N 500 --M 2 --full_basis 0 --dim 10 --sim_type orthogonal
# dim 15
# python src/main.py --single_test 0 --num_coeffs_change 2 --train_percent 0.4 --window_size 100 --step_size 1 --sim 1 --lam 3e-1 --N 500 --M 4 --full_basis 0 --dim 15 --sim_type cholesky
# python src/main.py --single_test 0 --num_coeffs_change 2 --train_percent 0.4 --window_size 100 --step_size 1 --sim 1 --lam 3e-1 --N 500 --M 3 --full_basis 0 --dim 15 --sim_type orthogonal
# dim 20
# python src/main.py --single_test 0 --num_coeffs_change 2 --train_percent 0.4 --window_size 100 --step_size 1 --sim 1 --lam 3e-1 --N 500 --M 6 --full_basis 0 --dim 20 --sim_type cholesky
# python src/main.py --single_test 0 --num_coeffs_change 2 --train_percent 0.4 --window_size 100 --step_size 1 --sim 1 --lam 3e-1 --N 500 --M 4 --full_basis 0 --dim 20 --sim_type orthogonal
# dim 30
# python src/main.py --single_test 0 --num_coeffs_change 2 --train_percent 0.4 --window_size 100 --step_size 1 --sim 1 --lam 3e-1 --N 500 --M 8 --full_basis 0 --dim 30 --sim_type cholesky
# python src/main.py --single_test 0 --num_coeffs_change 2 --train_percent 0.4 --window_size 100 --step_size 1 --sim 1 --lam 3e-1 --N 500 --M 5 --full_basis 0 --dim 30 --sim_type orthogonal
# dim 50
# python src/main.py --sim 1 --sim_type orthogonal_mult_coeff --dim 50 --window_size 100 --step_size 1 --lam 1e-1 --single_test 0 --N 500 --M 10 --t 1.0 --train_percent 0.4 --full_basis 1 --beta 6e-3 --num_coeffs_change 2
# python src/main.py --sim 1 --sim_type cholesky --dim 50 --window_size 100 --step_size 1 --lam 3e-1 --single_test 0 --N 500 --M 10 --t 1.0 --train_percent 0.45 --full_basis 1 --beta 6e-3 --num_coeffs_change 2
# dim 100
# python src/main.py --single_test 0 --num_coeffs_change 2 --train_percent 0.4 --window_size 100 --step_size 1 --sim 1 --lam 3e-1 --N 300 --M 15 --full_basis 0 --dim 100 --sim_type cholesky
# python src/main.py --single_test 0 --num_coeffs_change 2 --train_percent 0.4 --window_size 100 --step_size 1 --sim 1 --lam 3e-1 --N 300 --M 10 --full_basis 0 --dim 100 --sim_type orthogonal
# --results_path /nfs/hpc/share/dinkinst/Correlation-Changepoint-Detection/results



