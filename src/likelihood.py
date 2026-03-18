import numpy as np
from tqdm import tqdm
from optim import optimize_coeffs, optimize_single_coeff, optimize_coeffs_first_order, optimize_coeffs_first_order_single
from optim import create_all_optim_problems, create_global_problem
from optim import create_all_optim_problems_cluster, solve_optim_single_cluster
from optim import solve_optim_single, solve_optim_global, coord_ascent, optim_boyd, optim_boyd_dc
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

import copy


def likelihood_ratio_test(likelihood_null, likelihood_alternative, dof, log_pvals=None):
    delta_d = -2*(likelihood_null-likelihood_alternative)
    pval = chi2.sf(delta_d, dof)
    if log_pvals:
        """
        special case dof=2
        """
        pval = -delta_d/2
    return delta_d, pval

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
    psi_hat = np.sum(np.expand_dims(alphas, 1)*H_s, 0)
    psi_hat = symmetrize_from_vector(psi_hat, P)
    l1_penalty = 0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if i != j:
                l1_penalty += np.abs(psi_hat[i,j])
    
    if include_l1:
        _, first_term = np.linalg.slogdet(psi_hat)
        likelihood = first_term - np.trace(psi_hat@C) - lam*l1_penalty - P*np.log(2*np.pi)
        assert is_pos_def(psi_hat), pdb.set_trace()
        return likelihood*(N/2)
    
    else:
        first_term = fast_logdet(psi_hat)
        likelihood = first_term - np.trace(psi_hat@C) - P*np.log(2*np.pi)
        if not is_pos_def(psi_hat):
            print(debug_title)
        return likelihood*(N/2)
    
def cluster_likelihood(alpha, H, C, N, debug_title='global'):
    P = C.shape[0]
    psi_hat = alpha*H
    first_term = fast_logdet(psi_hat)
    likelihood = first_term - np.trace(psi_hat@C) - P*np.log(2*np.pi)

    return likelihood*(N/2)

def LRT_all_coeffs(data_total, M, dim, H_s, window_size=500, lam=1e-2, step_size=1):
    print("DEPRECATED FUNCTION - NOT USING FULL LIKELIHOOD LRT")
    exit(1)
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
    print("DEPRECATED FUNCTION - NOT USING FULL LIKELIHOOD LRT")
    exit(1)
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

def LRT_individual_coeffs_full_likelihood(data_total, M, dim, H_s, window_size=500, post_window_size=100, lam=1e-2, step_size=1, beta=5e-3, iters=150, include_l1=False, t=2.0, optim_type='unbiased', args=None):
    lrt_vals = []
    p_vals = []
    print("Creating CVX Problems...")
    prob_dict = create_all_optim_problems(H_s, dim=dim)
    print("Creating Cluster Specific CVX Problems...")
    prob_dict_clust = create_all_optim_problems_cluster(H_s, dim=dim)
    g_prob = create_global_problem(H_s, dim=dim)

    if args.fix_pre:
        print("----> Fixing Pre Window <----")
        data_one = data_total[:, 0:window_size]
        C_one = np.cov(data_one, bias=True)
        C_one = C_one + np.eye(C_one.shape[0])*1e-8 # correct numerical instability
        end_idx = data_total.shape[1]-post_window_size
    else: 
        print("---> NOT Fixing Pre Window <---")
        end_idx = data_total.shape[1]-2*window_size

    for i in tqdm(range(0, end_idx, step_size)): # this is still running over the training data, just needs adjustment afterward
        if args.fix_pre:
            data_two = data_total[:, i:i+post_window_size]
        else:
            data_one = data_total[:, i:i+window_size]
            C_one = np.cov(data_one, bias=True)
            C_one = C_one + np.eye(C_one.shape[0])*1e-8
            data_two = data_total[:, i+window_size:i+2*window_size]
        data_full = np.concatenate((data_one, data_two), axis=1)

        C_two = np.cov(data_two, bias=True)
        C_two = C_two + np.eye(C_two.shape[0])*1e-8 # correct numerical instability
        C_full = np.cov(data_full, bias=True)
        C_full = C_full + np.eye(C_full.shape[0])*1e-8
        

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
            coeffs_hat_total = unbiased_init_precision(C=C_full, H_s=H_s)
        elif optim_type == 'Boyd':
            coeffs_hat_total = optim_boyd(C=C_full, H_s=H_s)
        
        null_likelihood = full_likelihood(coeffs_hat_total, H_s, C_full, N=data_full.shape[1], 
                                          lam=lam, include_l1=include_l1, debug_title='global')

        test_stats_m = []
        p_vals_m = []

        for k in range(M):
            ####################################
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
                DEPRECATED
                """
                print("DEPRECATED OPTIMIZATION TYPE - NOT USING UNBIASED")
                exit(1)
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
                alpha_i_change_pre = coeffs_hat_total.copy()
                alpha_i_change_post = coeffs_hat_total.copy()
                curr_alpha_i_pre = optim_boyd_dc(C=C_one, H=H_s[k])
                curr_alpha_i_post = optim_boyd_dc(C=C_two, H=H_s[k])
                alpha_i_change_pre[k] = curr_alpha_i_pre
                alpha_i_change_post[k] = curr_alpha_i_post
            
            # # likelihood on pre data, alpha_one change
            alt_likelihood_alpha_i_pre = full_likelihood(alpha_i_change_pre, H_s, C_one, N=data_one.shape[1], 
                                                         lam=lam, include_l1=include_l1, debug_title='Pre')
            # likelihood on post data, alpha_one change
            alt_likelihood_alpha_i_post = full_likelihood(alpha_i_change_post, H_s, C_two, N=data_two.shape[1], 
                                                          lam=lam, include_l1=include_l1, debug_title='Post')
            

            # total likelihood alt first coeff
            alt_likelihood_alpha_i = alt_likelihood_alpha_i_pre + alt_likelihood_alpha_i_post
            dof = 2
            test_stat_i, p_val_i = likelihood_ratio_test(null_likelihood, 
                                                alt_likelihood_alpha_i, dof, log_pvals=args.log_pvals)            
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

def LRT_covariance(data_total, window_size=500, step_size=50):
    print('DEPRECATED FUNCTION - NOT USING COVARIANCE LRT')
    exit(1)
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

