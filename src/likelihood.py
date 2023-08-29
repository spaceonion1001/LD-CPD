import numpy as np
from tqdm import tqdm
from optim import optimize_coeffs, optimize_single_coeff, optimize_coeffs_first_order, optimize_coeffs_first_order_single
from optim import create_all_optim_problems, create_global_problem
from optim import create_all_optim_problems_cluster, solve_optim_single_cluster
from optim import solve_optim_single, solve_optim_global, coord_ascent, optim_boyd
from optim import iterative_soln_precision_single, iterative_soln_precision, unbiased_init_precision, unbiased_init_precision_single, unbiased_init_precision_single_alt
from statsmodels.stats.multitest import fdrcorrection, multipletests
from scipy.stats import chi2, multivariate_normal
from utils import is_pos_def, is_symmetric, vectorize_matrix, symmetrize_from_vector
from numba import jit
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from numpy.linalg import inv
from sklearn.utils.extmath import fast_logdet

import multiprocessing as mp
from multiprocessing import Pool
from itertools import repeat
from functools import partial
import copy

import dask
from joblib import Parallel, delayed

def likelihood_ratio_test(likelihood_null, likelihood_alternative, dof):
    delta_d = -2*(likelihood_null-likelihood_alternative)
    
    return delta_d, chi2.pdf(delta_d, dof)

def apply_fdr_correction(p_vals_all, alpha=0.05):
    corrected_p_vals_all = []
    for i in range(p_vals_all.shape[0]):
        rejected, corrected_p_vals_i = fdrcorrection(p_vals_all[i], alpha=alpha)
        corrected_p_vals_all.append(corrected_p_vals_i)
    return np.array(corrected_p_vals_all)

def apply_bonferroni_correction(p_vals_all, alpha=0.05):
    corrected_p_vals_all = []
    for i in range(p_vals_all.shape[0]):
        rejected, corrected_p_vals_i, _, _ = multipletests(p_vals_all[i], alpha=alpha, method='bonferroni')
        corrected_p_vals_all.append(corrected_p_vals_i)
    return np.array(corrected_p_vals_all)

# this is maximized likelihood, not NLL
def lasso_likelihood(alphas, H_s, C, lam=1e-2, include_l1=False):
    dim = C.shape[0]
    psi_hat = sum([alphas[i]*H_s[i] for i in range(H_s.shape[0])])
    psi_hat = symmetrize_from_vector(psi_hat, dim)
    l1_penalty = sum([np.abs(psi_hat[i, j])
                  for i in range(C.shape[0])
                  for j in range(C.shape[1]) if i != j])
    
    if include_l1:
        return np.log(np.linalg.det(psi_hat)) - np.trace(psi_hat@C) - lam*l1_penalty
    
    else:
        return np.log(np.linalg.det(psi_hat)) - np.trace(psi_hat@C)

#@jit(nopython=True)
def full_likelihood(alphas, H_s, C, N, lam=1e-2, include_l1=False, debug_title='global'):
    P = C.shape[0]
    #psi_hat = sum([alphas[i]*H_s[i] for i in range(H_s.shape[0])])
    psi_hat = np.sum(np.expand_dims(alphas, 1)*H_s, 0)
    psi_hat = symmetrize_from_vector(psi_hat, P)
    # l1_penalty = np.sum([np.abs(psi_hat[i, j])
    #               for i in range(C.shape[0])
    #               for j in range(C.shape[1]) if i != j])
    l1_penalty = 0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if i != j:
                l1_penalty += np.abs(psi_hat[i,j])
    
    if include_l1:
        _, first_term = np.linalg.slogdet(psi_hat)
        #likelihood = np.log(np.linalg.det(psi_hat)) - np.trace(psi_hat@C) - lam*l1_penalty - P*np.log(2*np.pi)
        likelihood = first_term - np.trace(psi_hat@C) - lam*l1_penalty - P*np.log(2*np.pi)
        assert is_pos_def(psi_hat), pdb.set_trace()
        return likelihood*(N/2)
    
    else:
        #_, first_term = np.linalg.slogdet(psi_hat)
        first_term = fast_logdet(psi_hat)
        #likelihood = np.log(np.linalg.det(psi_hat)) - np.trace(psi_hat@C) - P*np.log(2*np.pi)
        likelihood = first_term - np.trace(psi_hat@C) - P*np.log(2*np.pi)
        if not is_pos_def(psi_hat):
            print(debug_title)
            #pdb.set_trace()
        #assert is_pos_def(psi_hat), debug_title+str(alphas)
        return likelihood*(N/2)
    
def cluster_likelihood(alpha, H, C, N, debug_title='global'):
    P = C.shape[0]
    psi_hat = alpha*H
    _, first_term = np.linalg.slogdet(psi_hat)
    likelihood = first_term - np.trace(psi_hat@C) - P*np.log(2*np.pi)

    return likelihood*(N/2)

def LRT_all_coeffs(data_total, M, dim, H_s, window_size=500, lam=1e-2, step_size=1):
    lrt_vals = []
    p_vals = []
    for i in tqdm(range(0, data_total.shape[1]-2*window_size, step_size)):
        data_one = data_total[:, i:i+window_size]
        data_two = data_total[:, i+window_size:i+2*window_size]
        data_full = np.concatenate((data_one, data_two), axis=1)
        C_one = np.cov(data_one, bias=True)
        C_two = np.cov(data_two, bias=True)
        C_full = np.cov(data_full, bias=True)

        coeffs_hat_one = optimize_coeffs(H_s, C_one, lam=lam)
        coeffs_hat_two = optimize_coeffs(H_s, C_two, lam=lam)
        coeffs_hat_total = optimize_coeffs(H_s, C_full, lam=lam)
        null_first = lasso_likelihood(coeffs_hat_total, H_s, C_one, lam=lam)
        null_second = lasso_likelihood(coeffs_hat_total, H_s, C_two, lam=lam)
        null_likelihood = null_first + null_second
        alt_likelihood_one = lasso_likelihood(coeffs_hat_one, H_s, C_one, lam=lam)
        alt_likelihood_two = lasso_likelihood(coeffs_hat_two, H_s, C_two, lam=lam)
            
        test_stat, p_val = likelihood_ratio_test(null_likelihood, alt_likelihood_one+alt_likelihood_two, M)
        lrt_vals.append(test_stat)
        p_vals.append(p_val)
    return lrt_vals, p_vals

def LRT_all_coeffs_full_likelihood(data_total, M, dim, H_s, window_size=500, lam=1e-2, step_size=1, beta=5e-3, iters=100, include_l1=True, t=2.0):
    lrt_vals = []
    p_vals = []
    for i in tqdm(range(0, data_total.shape[1]-2*window_size, step_size)):
        data_one = data_total[:, i:i+window_size]
        data_two = data_total[:, i+window_size:i+2*window_size]
        data_full = np.concatenate((data_one, data_two), axis=1)
        C_one = np.cov(data_one, bias=True)
        C_two = np.cov(data_two, bias=True)
        C_full = np.cov(data_full, bias=True)

        ###############
        # CVX SOLVER #

        # coeffs_hat_one = optimize_coeffs(H_s, C_one, lam=lam)
        # coeffs_hat_two = optimize_coeffs(H_s, C_two, lam=lam)
        # coeffs_hat_total = optimize_coeffs(H_s, C_full, lam=lam)

        # FIRST ORDER SUBDERIV #
        coeffs_hat_one = optimize_coeffs_first_order(H_s, C_one, lam=lam, beta=beta, iters=iters, t=t)
        coeffs_hat_two = optimize_coeffs_first_order(H_s, C_two, lam=lam, beta=beta, iters=iters, t=t)
        coeffs_hat_total = optimize_coeffs_first_order(H_s, C_full, lam=lam, beta=beta, iters=iters, t=t)

        ################
        null_first = full_likelihood(coeffs_hat_total, H_s, C_one, N=data_one.shape[1], 
                                        lam=lam, include_l1=include_l1)
        null_second = full_likelihood(coeffs_hat_total, H_s, C_two, N=data_two.shape[1], 
                                        lam=lam, include_l1=include_l1)
        null_likelihood = null_first + null_second
        alt_likelihood_one = full_likelihood(coeffs_hat_one, H_s, C_one, N=data_one.shape[1], 
                                        lam=lam, include_l1=include_l1)
        alt_likelihood_two = full_likelihood(coeffs_hat_two, H_s, C_two, N=data_two.shape[1], 
                                        lam=lam, include_l1=include_l1)
            
        test_stat, p_val = likelihood_ratio_test(null_likelihood, alt_likelihood_one+alt_likelihood_two, M)
        lrt_vals.append(test_stat)
        p_vals.append(p_val)
    return lrt_vals, p_vals

def LRT_individual_coeffs_full_likelihood(data_total, M, dim, H_s, window_size=500, lam=1e-2, step_size=1, beta=5e-3, iters=150, include_l1=False, t=2.0, optim_type='unbiased'):
    lrt_vals = []
    p_vals = []
    print("Creating CVX Problems...")
    prob_dict = create_all_optim_problems(H_s, dim=dim)
    print("Creating Cluster Specific CVX Problems...")
    prob_dict_clust = create_all_optim_problems_cluster(H_s, dim=dim)
    g_prob = create_global_problem(H_s, dim=dim)
    data_one = data_total[:, 0:2*window_size]
    C_one = np.cov(data_one, bias=True)
    # alpha_i_change_pre = solve_optim_global(curr_C=C_one, g_prob=g_prob)
    # # likelihood on pre data, alpha_one change
    # alt_likelihood_alpha_i_pre = full_likelihood(alpha_i_change_pre, H_s, C_one, N=data_one.shape[1], 
    #                                                 lam=lam, include_l1=include_l1, debug_title='Pre')
    for i in tqdm(range(2*window_size, data_total.shape[1]-window_size, step_size)):
        start_win_indx = i
        first_end_indx = i+window_size
        last_end_indx = i+2*window_size
        data_two = data_total[:, i:i+window_size]
        #data_two = data_total[:, i+window_size:i+2*window_size]
        data_full = np.concatenate((data_one, data_two), axis=1)
        #C_one = np.cov(data_one, bias=True)
        C_two = np.cov(data_two, bias=True)
        C_full = np.cov(data_full, bias=True)
        

        #############################
        coeffs_hat_total = np.ones(M)
        if optim_type == 'CVX':
            coeffs_hat_total = solve_optim_global(curr_C=C_full, g_prob=g_prob)
        elif optim_type == 'CXVCLUST':
            for k in range(M):
                curr_H = H_s[k]
                curr_H = symmetrize_from_vector(curr_H, dim=dim)
                curr_C_pre = C_one[~np.all(curr_H==0, axis=1), :][:, ~np.all(curr_H==0, axis=0)]
                curr_C_post = C_two[~np.all(curr_H==0, axis=1), :][:, ~np.all(curr_H==0, axis=0)]
                coeffs_hat_total = solve_optim_single_cluster(curr_alphas=coeffs_hat_total, 
                                                                curr_C=curr_C_pre,
                                                                prob_dict=prob_dict_clust,
                                                                optim_idx=k
                                                                )
                coeffs_hat_total = solve_optim_single_cluster(curr_alphas=coeffs_hat_total, 
                                                                curr_C=curr_C_post,
                                                                prob_dict=prob_dict_clust,
                                                                optim_idx=k
                                                                )
        elif optim_type == 'unbiased':
            coeffs_hat_total = unbiased_init_precision(C=C_full, H_s=H_s)
        elif optim_type == 'Anderson':
            coeffs_hat_total = iterative_soln_precision(coeffs_zero=np.ones(H_s.shape[0]), H_s=H_s, C=C_full, iters=15)
        elif optim_type == 'first-order':
            #coeffs_hat_total = optimize_coeffs_first_order(H_s, C_full, lam=lam, beta=beta, iters=iters, t=t)
            #coeffs_hat_total = solve_optim_global(curr_C=C_full, g_prob=g_prob)
            coeffs_hat_total = unbiased_init_precision(C=C_full, H_s=H_s)
        elif optim_type == 'Boyd':
            coeffs_hat_total = optim_boyd(C=C_full, H_s=H_s)
        #coeffs_hat_total = optimize_coeffs(H_s, C_full, lam=lam)
        # psi_hat_temp = np.sum(np.expand_dims(coeffs_hat_total, 1)*H_s, 0)
        # psi_hat_temp = symmetrize_from_vector(psi_hat_temp, dim)
        
        #print("COEFFS HAT TOTAL", coeffs_hat_total)
        ##############################
        # null likelihood
        # TODO
        # null_first = full_likelihood(coeffs_hat_total, H_s, C_one, N=data_one.shape[1], 
        #                              lam=lam, include_l1=include_l1)
        # null_second = full_likelihood(coeffs_hat_total, H_s, C_two, N=data_two.shape[1], 
        #                              lam=lam, include_l1=include_l1)
        #null_likelihood = null_first + null_second
        null_likelihood = full_likelihood(coeffs_hat_total, H_s, C_full, N=data_full.shape[1], 
                                          lam=lam, include_l1=include_l1, debug_title='global')
        # TODO

        test_stats_m = []
        p_vals_m = []
        # parallel optimization
        N_cpus = 6
        k_vals = np.arange(0, M)
        #TODO
        dask_list_pre = []
        dask_list_post = []
        #pre_results = Parallel(n_jobs=4, prefer='threads')(delayed(solve_optim_single)(val, coeffs_hat_total, C_one, prob_dict) for val in k_vals)
        #post_results = Parallel(n_jobs=4, prefer='threads')(delayed(solve_optim_single)(val, coeffs_hat_total, C_two, prob_dict) for val in k_vals)
        # for val in k_vals:
        #     dask_list_pre.append(dask.delayed(solve_optim_single)(val, coeffs_hat_total, C_one, copy.copy(prob_dict)))
        #     dask_list_post.append(dask.delayed(solve_optim_single)(val, coeffs_hat_total, C_two, copy.copy(prob_dict)))
        # pre_results = dask.compute(*dask_list_pre, scheduler='threads', num_workers=N_cpus)
        # post_results = dask.compute(*dask_list_post, scheduler='threads', num_workers=N_cpus)
        #TODO
        # iterate over each coefficient
        for k in range(M):
            ####################################
            #alpha_i_change_pre = np.ones(M)
            #alpha_i_change_post = np.ones(M)
            if optim_type == 'CVX':
                alpha_i_change_pre = solve_optim_single(curr_alphas=coeffs_hat_total, 
                                                            curr_C=C_one,
                                                            prob_dict=prob_dict,
                                                            optim_idx=k)
                alpha_i_change_post = solve_optim_single(curr_alphas=coeffs_hat_total, 
                                                                curr_C=C_two,
                                                                prob_dict=prob_dict,
                                                                optim_idx=k)
            elif optim_type == 'CVXCLUST':
                curr_H = H_s[k]
                curr_H = symmetrize_from_vector(curr_H, dim=dim)
                curr_C_pre = C_one[~np.all(curr_H==0, axis=1), :][:, ~np.all(curr_H==0, axis=0)]
                curr_C_post = C_two[~np.all(curr_H==0, axis=1), :][:, ~np.all(curr_H==0, axis=0)]
                alpha_i_change_pre = solve_optim_single_cluster(curr_alphas=coeffs_hat_total, 
                                                                curr_C=curr_C_pre,
                                                                prob_dict=prob_dict_clust,
                                                                optim_idx=k
                                                                )
                alpha_i_change_post = solve_optim_single_cluster(curr_alphas=coeffs_hat_total, 
                                                                curr_C=curr_C_post,
                                                                prob_dict=prob_dict_clust,
                                                                optim_idx=k
                                                                )
                
            elif optim_type == 'unbiased':
                """
                THIS ONE PROBABLY ISN'T COMPLETE YET

                NEED CONVINCING THAT THIS IS OKAY TO THROW OUT INFO FROM OTHER CLUSTERS

                ADDITIONALLY, THIS CURRENTLY GIVES NEGATIVES IN THE LIKELIHOOD RATIO TEST
                WHICH IS A CONSEQUENCE OF SOLVING ALL COEFFICIENTS TOGETHER, BUT TOSSING OUT THE OTHERS
                (FIXING THEM TO BE THE VALUES FROM THE TOTAL, SINCE UNBIASED ASSUMES THEY ALL FLEX TOGETHER)
                """
                # alpha_i_change_pre = unbiased_init_precision_single(coeffs_zero=coeffs_hat_total,
                #                                                     C=C_one,
                #                                                     H_s=H_s,
                #                                                     modify_index=k
                #                                                     )
                # alpha_i_change_post = unbiased_init_precision_single(coeffs_zero=coeffs_hat_total,
                #                                                     C=C_two,
                #                                                     H_s=H_s,
                #                                                     modify_index=k
                #                                                     )
                alpha_i_change_pre = coord_ascent(coeffs_zero=coeffs_hat_total,
                                                                    C=C_one,
                                                                    H_s=H_s,
                                                                    modify_index=k,
                                                                    iters=iters,
                                                                    beta=beta
                                                                    )
                alpha_i_change_post = coord_ascent(coeffs_zero=coeffs_hat_total,
                                                                    C=C_two,
                                                                    H_s=H_s,
                                                                    modify_index=k,
                                                                    iters=iters,
                                                                    beta=beta
                                                                    )
            elif optim_type == 'Anderson':
                alpha_i_change_pre = iterative_soln_precision_single(
                                    coeffs_zero=coeffs_hat_total,
                                    H_s=H_s,
                                    C=C_one,
                                    dim=data_full.shape[0],
                                    modify_index=k,
                                    iters=10
                                )
                alpha_i_change_post = iterative_soln_precision_single(
                                        coeffs_zero=coeffs_hat_total,
                                        H_s=H_s,
                                        C=C_two,
                                        dim=data_full.shape[0],
                                        modify_index=k,
                                        iters=10
                            
                    )
            elif optim_type == 'first-order':
                #alpha_i_change_pre = optimize_coeffs_first_order_single(coeffs_hat_total, H_s, C_one, lam=lam, beta=beta, iters=iters, optim_indx=k, t=t)
                #alpha_i_change_post = optimize_coeffs_first_order_single(coeffs_hat_total, H_s, C_two, lam=lam, beta=beta, iters=iters, optim_indx=k, t=t)
                alpha_i_change_pre = coord_ascent(coeffs_zero=coeffs_hat_total,
                                                                    C=C_one,
                                                                    H_s=H_s,
                                                                    modify_index=k,
                                                                    iters=iters,
                                                                    beta=beta
                                                                    )
                alpha_i_change_post = coord_ascent(coeffs_zero=coeffs_hat_total,
                                                                    C=C_two,
                                                                    H_s=H_s,
                                                                    modify_index=k,
                                                                    iters=iters,
                                                                    beta=beta
                                                                    )
            elif optim_type == 'Boyd':
                alpha_i_change_pre = optim_boyd(C=C_one, H_s=H_s)
                alpha_i_change_post = optim_boyd(C=C_two, H_s=H_s)
            # alpha_i_change_pre = optimize_single_coeff(coeffs_hat_total, H_s, C_one, coeff_idx=i,
            #                                            lam=lam)
            # alpha_i_change_post = optimize_single_coeff(coeffs_hat_total, H_s, C_two, coeff_idx=i,
            #                                             lam=lam)
            
            # #TODO
            # psi_hat_temp = np.sum(np.expand_dims(alpha_i_change_pre, 1)*H_s, 0)
            # psi_hat_temp = symmetrize_from_vector(psi_hat_temp, dim)
            # if not is_pos_def(psi_hat_temp):
            #     alpha_i_change_pre = solve_optim_single(curr_alphas=coeffs_hat_total, 
            #                                                 curr_C=C_one,
            #                                                 prob_dict=prob_dict,
            #                                                 optim_idx=k)
            # psi_hat_temp = np.sum(np.expand_dims(alpha_i_change_post, 1)*H_s, 0)
            # psi_hat_temp = symmetrize_from_vector(psi_hat_temp, dim)
            # if not is_pos_def(psi_hat_temp):
            #     alpha_i_change_post = solve_optim_single(curr_alphas=coeffs_hat_total, 
            #                                                 curr_C=C_two,
            #                                                 prob_dict=prob_dict,
            #                                                 optim_idx=k)
            
            
            #TODO
            # print("*** GLOBAL ***")
            # print(np.around(coeffs_hat_total, decimals=3))
            # print("**************")
            # print("\n*** PRE ***")
            # print("Ours", np.around(alpha_i_change_pre, decimals=3))
            # print("Solver", np.around(alpha_i_change_pre_alt, decimals=3))
            # print("Dist {}".format(np.around(np.linalg.norm(alpha_i_change_pre-alpha_i_change_pre_alt, ord=1), decimals=3)))
            # dist_global_ours_pre = np.around(np.linalg.norm(coeffs_hat_total-alpha_i_change_pre, ord=1), decimals=3)
            # dist_global_solver_pre = np.around(np.linalg.norm(coeffs_hat_total-alpha_i_change_pre_alt, ord=1), decimals=3)
            # print("Dist Global Ours {}".format(dist_global_ours_pre))
            # print("Dist Global Solver {}".format(dist_global_solver_pre))
            # print("***********")
            # print("\n*** Post ***")
            # print("Ours", np.around(alpha_i_change_post, decimals=3))
            # print("Solver", np.around(alpha_i_change_post_alt, decimals=3))
            # print("Dist {}".format(np.around(np.linalg.norm(alpha_i_change_post-alpha_i_change_post_alt, ord=1), decimals=3)))
            # dist_global_ours_post = np.around(np.linalg.norm(coeffs_hat_total-alpha_i_change_post, ord=1), decimals=3)
            # dist_global_solver_post = np.around(np.linalg.norm(coeffs_hat_total-alpha_i_change_post_alt, ord=1), decimals=3)
            # print("Dist Global Ours {}".format(dist_global_ours_post))
            # print("Dist Global Solver {}".format(dist_global_solver_post))
            # print("***********\n\n")


            
            #TODO
            
            # # likelihood on pre data, alpha_one change
            alt_likelihood_alpha_i_pre = full_likelihood(alpha_i_change_pre, H_s, C_one, N=data_one.shape[1], 
                                                         lam=lam, include_l1=include_l1, debug_title='Pre')
            # likelihood on post data, alpha_one change
            alt_likelihood_alpha_i_post = full_likelihood(alpha_i_change_post, H_s, C_two, N=data_two.shape[1], 
                                                          lam=lam, include_l1=include_l1, debug_title='Post')
            
            """
            For cluster specific likelihood
            """
            # if optim_type == 'CVXCLUST':
            #     curr_H = H_s[k]
            #     curr_H = symmetrize_from_vector(curr_H, dim=dim)
            #     curr_C_pre = C_one[~np.all(curr_H==0, axis=1), :][:, ~np.all(curr_H==0, axis=0)]
            #     curr_C_post = C_two[~np.all(curr_H==0, axis=1), :][:, ~np.all(curr_H==0, axis=0)]
            #     curr_C_full = C_full[~np.all(curr_H==0, axis=1), :][:, ~np.all(curr_H==0, axis=0)]
            #     curr_H = curr_H[~np.all(curr_H==0, axis=1), :][:, ~np.all(curr_H==0, axis=0)]
            #     alt_likelihood_alpha_i_pre = cluster_likelihood(alpha_i_change_pre[k],
            #                                                     curr_H,
            #                                                     curr_C_pre,
            #                                                     N=data_one.shape[1],
            #                                                     debug_title='Pre'
            #                                                     )
            #     alt_likelihood_alpha_i_post = cluster_likelihood(alpha_i_change_post[k],
            #                                                     curr_H,
            #                                                     curr_C_post,
            #                                                     N=data_two.shape[1],
            #                                                     debug_title='Post'
            #                                                     )
            #     null_likelihood = cluster_likelihood(coeffs_hat_total[k],
            #                                         curr_H,
            #                                         curr_C_full,
            #                                         N=data_total.shape[1],
            #                                         debug_title='Global'
            #                                         )

            """
            END
            """
            
            # alt_likelihood_alpha_i_pre_alt = full_likelihood(alpha_i_change_pre_alt, H_s, C_one, N=data_one.shape[1], 
            #                                              lam=lam, include_l1=include_l1, debug_title='Pre')
            # # likelihood on post data, alpha_one change
            # alt_likelihood_alpha_i_post_alt = full_likelihood(alpha_i_change_post_alt, H_s, C_two, N=data_two.shape[1], 
            #                                               lam=lam, include_l1=include_l1, debug_title='Post')
            #print("Likelihood Diff Pre {}".format(alt_likelihood_alpha_i_pre-alt_likelihood_alpha_i_pre_alt))
            #print("Likelihood Diff Post {}".format(alt_likelihood_alpha_i_post-alt_likelihood_alpha_i_post_alt))
            #TODO
            # likelihood on pre data, alpha_one change
            # alt_likelihood_alpha_i_pre = full_likelihood(pre_results[k], H_s, C_one, N=data_one.shape[1], 
            #                                              lam=lam, include_l1=include_l1)
            # # likelihood on post data, alpha_one change
            # alt_likelihood_alpha_i_post = full_likelihood(post_results[k], H_s, C_two, N=data_two.shape[1], 
            #                                               lam=lam, include_l1=include_l1)

            # total likelihood alt first coeff
            alt_likelihood_alpha_i = alt_likelihood_alpha_i_pre + alt_likelihood_alpha_i_post
            #alt_likelihood_alpha_i_alt = alt_likelihood_alpha_i_pre_alt + alt_likelihood_alpha_i_post_alt
            #print("Likelihood Diff Total {}".format(alt_likelihood_alpha_i - alt_likelihood_alpha_i_alt))
            
            test_stat_i, p_val_i = likelihood_ratio_test(null_likelihood, 
                                                alt_likelihood_alpha_i, 2)
            # test_stat_i_alt, p_val_i_alt = likelihood_ratio_test(null_likelihood, 
            #                                     alt_likelihood_alpha_i_alt, 2)
            #print("\n\n")
            
            test_stats_m.append(test_stat_i)
            p_vals_m.append(p_val_i)
            
        lrt_vals.append(test_stats_m)
        p_vals.append(p_vals_m)
    
    # return arrays with shape (num_tests, M)
    return np.array(lrt_vals), np.array(p_vals)


def calc_likelihood_covariance(data, C):
    assert is_pos_def(C), "Not PosDef"
    assert is_symmetric(C), "Not Symmetric"
    log_l = multivariate_normal.logpdf(data.T, mean=np.zeros(data.shape[0]), cov=C).sum()
    
    return log_l
    #return log_l*(N/2)

def LRT_covariance(data_total, window_size=500, step_size=50):
    lrt_vals = []
    p_vals = []
    null_likelihoods = []
    alt_likelihoods = []
    dim = data_total.shape[0]
    for i in tqdm(range(0, data_total.shape[1]-2*window_size, step_size)):
        data_one = data_total[:, i:i+window_size]
        data_two = data_total[:, i+window_size:i+2*window_size]
        data_full = np.concatenate((data_one, data_two), axis=1)
        C_one = np.cov(data_one, bias=True)
        C_two = np.cov(data_two, bias=True)
        C_full = np.cov(data_full, bias=True)
        # null likelihood
        null_first = calc_likelihood_covariance(data_one, C_full)
        null_second = calc_likelihood_covariance(data_two, C_full)
        null_likelihood = null_first + null_second
        alt_likelihood_one = calc_likelihood_covariance(data_one, C_one)
        alt_likelihood_two = calc_likelihood_covariance(data_two, C_two)
        
        dof_calc = (dim*(dim+1))/2
        test_stat, p_val = likelihood_ratio_test(null_likelihood, alt_likelihood_one+alt_likelihood_two, 
                                                 dof_calc)
        lrt_vals.append(test_stat)
        p_vals.append(p_val)
        null_likelihoods.append(null_likelihood)
        alt_likelihoods.append(alt_likelihood_one+alt_likelihood_two)

    return np.array(lrt_vals), np.array(p_vals), np.array(null_likelihoods), np.array(alt_likelihoods)

