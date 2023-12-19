import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import statistics
import pickle
from math import sqrt
matplotlib.use("Agg")
sns.set()

from sklearn.covariance import GraphicalLasso
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2, multivariate_normal
from sklearn.metrics import silhouette_score, pairwise_distances as pairwise_d
from sklearn.utils.extmath import fast_logdet


from tqdm import tqdm
from simulate import generate_matrices_orthogonal, sim_data, collect_precision_matrix
from likelihood import full_likelihood, likelihood_ratio_test
from numpy.linalg import inv as inv
from optim import optim_boyd, optim_boyd_dc
from utils import symmetrize_from_vector, vectorize_matrix, is_pos_def, scale_data
np.random.seed(42)
import argparse

import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
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
from kesh_cpd import calc_T_t, calc_g1, calc_g2, calc_rhat

from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.cluster.hierarchy as sch
import scipy

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lam', type=float, default=5e-2)
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--dim', type=int, default=80)
    parser.add_argument('--w', type=int, default=100)
    parser.add_argument('--prec_rec', type=int, default=0)
    parser.add_argument('--load_results', type=int, default=0)
    args = parser.parse_args()

    return args

def plot_confidence_interval(x, values, z=1.96, color='#2187bb', horizontal_line_width=0.25, constrain=False):
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    confidence_interval = z * stdev / sqrt(len(values))

    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval
    if constrain: # for pvalue plots
        if top <= 0.0:
            top = 0.0
    plt.plot([x, x], [top, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)
    plt.plot(x, mean, 'o', color='#f44336')

    return mean, confidence_interval

def clime_init_fn(lam, data_minimal):
    nrow, ncol = data_minimal.shape
    X = r.matrix(data_minimal, nrow=nrow, ncol=ncol)
    reg_soln = scalreg.scalreg(X, lam0=lam)
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

def anderson_lrt(cluster_precision, C, N):
    """
    Anderson LRT for goodness-of-fit

    -2*log of 4.10

    -Nlogdet(C)-Nlogdet(precision)
    """
    
    # calculate likelihood ratio criterion eq 4.10
    #print("C {} Prec {}".format(C.shape, cluster_precision.shape))
    first_term = 0.5*N*fast_logdet(C)
    second_term = 0.5*N*fast_logdet(cluster_precision)
    res = -2*(first_term + second_term)
    #print("First Term {} Second Term {} Res {}".format(first_term, second_term, res))

    return res

def recursive_split(basis_mat, dim):
    performed_split = True
    orig_nonzero_cols = np.nonzero(np.any(basis_mat != 0, axis=0))[0]
    basis_mat_reduced = basis_mat[:, ~np.all(basis_mat == 0, axis=0)]
    basis_mat_reduced = basis_mat_reduced[~np.all(basis_mat_reduced == 0, axis=1), :]
    #clust_dist_mat = np.abs(basis_mat)
    clust_dist_mat = basis_mat_reduced
    np.fill_diagonal(clust_dist_mat, 0.0)
    ###########
    clust_dist_mat = (clust_dist_mat.max()+1e-5) - clust_dist_mat
    np.fill_diagonal(clust_dist_mat, 0.0)
    ###########
    pairwise_distances = sch.distance.pdist(clust_dist_mat)
    Z = linkage(pairwise_distances, method='complete')
    # BREAK INTO 2 NEW CLUSTERS
    cutree = hierarchy.cut_tree(Z, n_clusters=2).squeeze()
    #print(np.nonzero(np.any(basis_mat != 0, axis=0))[0])
    #print(cutree)
    #print(len(np.nonzero(np.any(basis_mat != 0, axis=0))[0]))
    #print(len(cutree))
    new_basis_matrices = []
    for i in range(min(set(cutree)), max(set(cutree))+1): # iterate over clusters
        idxs = np.where(cutree == i)[0] # indexes for given cluster
        A = np.zeros(basis_mat.shape) # blank A matrix
        for idx in idxs: # loop over indexes
            for idx2 in idxs: # loop over indexes
                first_idx = orig_nonzero_cols[idx] # remap back to original space
                second_idx = orig_nonzero_cols[idx2] # remap back to original space
                A[first_idx][second_idx] = basis_mat[first_idx][second_idx].copy() # set i,j entry to be the entry from precision matrix for given cluster
        if len(np.nonzero(A)[0]) > 0: # checking for 0 entries
            new_basis_matrices.append(vectorize_matrix(A))
    new_basis_matrices = np.array(new_basis_matrices)
    if new_basis_matrices.shape[0] <= 1:
        #print("...Recursion Complete for Dendrogram Criterion...")
        performed_split = False
        return new_basis_matrices, performed_split
    
    for i in range(new_basis_matrices.shape[0]):
        curr_mat = symmetrize_from_vector(new_basis_matrices[i], dim)
        nonzero_cols = np.nonzero(np.any(curr_mat != 0, axis=0))[0]
        #print(nonzero_cols)
        #print("NONZERO COLS {}".format(nonzero_cols))
        if len(nonzero_cols) < 2: # don't split into a singleton cluster - not informative - 3x3 is the minimum for splitting
            #print("...Recursion Complete for Cluster Size {}...".format(len(nonzero_cols)))
            performed_split = False
            pass
        #print("Matrix {} Len {} Channels Contained {}".format(i, len(nonzero_cols), nonzero_cols))
        #print()
    #print("NEW BASIS MATRICES SHAPE", new_basis_matrices.shape)

    
    # if new_basis_matrices.shape[0] <= 1:
    #     performed_split = False

    return new_basis_matrices, performed_split

def bfs_basis_mats(data_train, curr_basis_mats, curr_silhoutte_score, dim, recursion_min, cutree, root_dist_mat):
    #print("###################################### RECURSION BLOCK LEVEL ######################################")
    C_full = np.cov(data_train.copy(), bias=True)
    level_basis_mats = []
    #g_prob = create_global_problem(curr_basis_mats, dim=self.dim)
    #alphas = solve_optim_global(curr_C=C_full, g_prob=g_prob)
    for i, mat in enumerate(curr_basis_mats):
        #print("TRAIN DATA {}".format(data_train.shape))
        orig_mat = mat.copy()
        mat = symmetrize_from_vector(mat, dim=dim)
        # if mat.shape[0] <= 4: # suitable localization - treat 4x4 or less as leaf nodes -> keep original matrix instead
        #     level_basis_mats.append(orig_mat)
        #     continue
        nonzero_cols = np.nonzero(np.any(mat != 0, axis=0))[0]
        if len(nonzero_cols) <= recursion_min:
            #print("...Recursion Complete for Singleton Cluster...")
            level_basis_mats.append(orig_mat)
            continue
        #print("NONZERO COLS {}".format(nonzero_cols))
        data_train_curr = data_train[nonzero_cols, :]
        #print("Data Train Curr {}".format(data_train_curr.shape))
        train_C = np.cov(data_train_curr.copy(), bias=True)
        #print("Train C {}".format(train_C.shape))
        train_C = train_C + np.eye(train_C.shape[0])*1e-8

        # fit current level of recursion on training data
        alphas = optim_boyd(C=C_full, H_s=curr_basis_mats)
        #print("ALPHAS BOYD {}".format(alphas))
        #print("Train C Shape {}".format(train_C.shape))
        #print("Alphas {}".format(alphas))
        cluster_precision = mat[~np.all(mat == 0, axis=1)]
        cluster_precision = alphas[i]*cluster_precision[:, ~np.all(cluster_precision == 0, axis=0)]
        anderson_lrt_value = anderson_lrt(cluster_precision=cluster_precision, C=train_C, N=data_train.shape[1])
        dof = 0.5*train_C.shape[0]*(train_C.shape[0]+1) - 1 # q here is just 1 since we are cluster specific
        chisquare_val = chi2.sf(anderson_lrt_value, dof)
        #print("CHISQUARE P-VAL {} DOF {}".format(chisquare_val, dof))
        new_mats, performed_split = recursive_split(mat, dim=dim)
        #print(new_mats.shape, performed_split)
        if performed_split:
            #print("BASIS MATS SHAPE {}".format(curr_basis_mats.shape))
            reduced_basis_mats = np.delete(curr_basis_mats, i, axis=0)
            #print("REDUCED MATS SHAPE {}".format(reduced_basis_mats.shape))
            updated_basis_matrices = np.concatenate((reduced_basis_mats, new_mats), axis=0)
            #print("UPDATED MATS SHAPE {}".format(updated_basis_matrices.shape))
            nonzero_cols_one = np.nonzero(np.any(symmetrize_from_vector(updated_basis_matrices[-2], dim) != 0, axis=0))[0]
            nonzero_cols_two = np.nonzero(np.any(symmetrize_from_vector(updated_basis_matrices[-1], dim) != 0, axis=0))[0]
            #print("***********************\nTesting Split of {}\nInto {}\nand {}\n***********************".format(nonzero_cols, nonzero_cols_one, nonzero_cols_two))
            silhoutte_labels = cutree
            silhoutte_labels[nonzero_cols_two] = int(cutree.max() + 1)
            new_silhoutte_score = silhouette_score(root_dist_mat, silhoutte_labels, metric='precomputed')
            #print("SILHOUETTE DIFFERENCE CURR {} NEW {}".format(curr_silhoutte_score, new_silhoutte_score))
            #print("CHISQUARE VAL {}".format(chisquare_val))
            #print("Silhoutte Scores", new_silhoutte_score, self.curr_silhoutte_score)
            condition_one = len(nonzero_cols_one) >= recursion_min and len(nonzero_cols_two) >= recursion_min
            condition_two = new_silhoutte_score > curr_silhoutte_score
            condition_three = chisquare_val >= 1e-5
            

            if condition_one and (condition_two or condition_three): # we split and the scores improved or didn't degrade (p-value) -> take new 2 matrices!
                #print("ADDING BASIS MATS")
                #print("NEW MATS SHAPE", new_mats.shape)
                curr_silhoutte_score = new_silhoutte_score
                for matr in new_mats:
                    level_basis_mats.append(matr)
            else: # both of these are false -> keep original matrix instead
                #print("...Recursion Complete for Fit-Check and Silhouette Score...")
                level_basis_mats.append(orig_mat)
        else: # split didn't take place -> keep original matrix instead
            level_basis_mats.append(orig_mat)
    #print(np.concatenate(level_basis_mats, axis=1).shape)
    level_basis_mats = np.array(level_basis_mats)
    if level_basis_mats.shape[0] > curr_basis_mats.shape[0]: # splits took place
        #print("$$$$$$$$$$$$$ RECURSING $$$$$$$$$$$$$$")
        #print("###################################### END OF RECURSION BLOCK LEVEL ######################################")
        level_basis_mats = bfs_basis_mats(data_train, level_basis_mats, curr_silhoutte_score, dim, recursion_min, silhoutte_labels, root_dist_mat) # recurse down a level


    return level_basis_mats


def collect_test_results(M, dim, w):
    # collect results
    seed_list = np.arange(0, 100)
    coeff_change_mags = np.arange(0.0, 1.0, 0.1)
    total_results = {}
    H_s = generate_matrices_orthogonal(M=M, dim=dim, to_print=False)[0]
    prec_one = collect_precision_matrix(H_s=H_s, prec_coeffs=np.ones(M), P=dim)
    for curr_change_mag in coeff_change_mags:
        coeff_diffs_curr = []
        coeff_diffs_curr_est = []
        lrt_vals_curr = []
        lrt_vals_true_curr = []
        lrt_vals_curr_est = []
        p_vals_curr = []
        p_vals_true_curr = []
        p_vals_curr_est = []
        test_vals_kesh_curr = []
        test_vals_kesh_true_curr = []
        mag_results = {}
        for curr_seed in tqdm(seed_list):
            second_prec_coeffs = np.ones(M)
            second_prec_coeffs[0] = second_prec_coeffs[0] - curr_change_mag
            prec_two = collect_precision_matrix(H_s=H_s, prec_coeffs=second_prec_coeffs, P=dim)
            data_one, _ = sim_data(covar=inv(prec_one), dim=dim, N=100)
            data_two, _ = sim_data(covar=inv(prec_two), dim=dim, N=100)
            data_full = np.concatenate((data_one, data_two), axis=1)
            C_one = np.cov(data_one, bias=True)
            C_two = np.cov(data_two, bias=True)
            C_full = np.cov(data_full, bias=True)
            coeffs_hat_total = optim_boyd(C=C_full, H_s=H_s)
            alpha_i_change_pre = coeffs_hat_total.copy()
            alpha_i_change_post = coeffs_hat_total.copy()
            curr_alpha_i_pre = optim_boyd_dc(C=C_one, H=H_s[0])
            curr_alpha_i_post = optim_boyd_dc(C=C_two, H=H_s[0])
            alpha_i_change_pre[0] = curr_alpha_i_pre
            alpha_i_change_post[0] = curr_alpha_i_post
            null_likelihood = full_likelihood(coeffs_hat_total, H_s, C_full, N=data_full.shape[1], 
                                                lam=5e-2, include_l1=False, debug_title='global')
            # # likelihood on pre data, alpha_one change
            alt_likelihood_alpha_i_pre = full_likelihood(alpha_i_change_pre, H_s, C_one, N=data_one.shape[1], 
                                                            lam=5e-2, include_l1=False, debug_title='Pre')
            # likelihood on post data, alpha_one change
            alt_likelihood_alpha_i_post = full_likelihood(alpha_i_change_post, H_s, C_two, N=data_two.shape[1], 
                                                            lam=5e-2, include_l1=False, debug_title='Post')
            alt_likelihood_alpha_i = alt_likelihood_alpha_i_pre + alt_likelihood_alpha_i_post
            dof = 2
            test_stat_i, p_val_i = likelihood_ratio_test(null_likelihood, 
                                                alt_likelihood_alpha_i, dof, log_pvals=1)
            curr_mat = symmetrize_from_vector(H_s[0], dim)
            nonzero_cols = np.nonzero(np.any(curr_mat != 0, axis=0))[0]
            C_val = len(nonzero_cols)
            if C_val > 0:
                correction_factor = dim/C_val
                #print(correction_factor)
                # correct p_vals for cluster at all time points independently
                p_val_i = p_val_i*correction_factor
            else:
                p_val_i = 1.0
            p_vals_curr.append(p_val_i)
            lrt_vals_curr.append(test_stat_i)
            coeff_diffs_curr.append(alpha_i_change_pre[0]-alpha_i_change_post[0])

            """"""
            # true LRT
            null_likelihood_true = full_likelihood(coeffs_hat_total, H_s, C_full, N=data_full.shape[1], 
                                                lam=5e-2, include_l1=False, debug_title='global')
            # # likelihood on pre data, alpha_one change
            alt_likelihood_alpha_i_pre_true = full_likelihood(np.ones(M), H_s, C_one, N=data_one.shape[1], 
                                                            lam=5e-2, include_l1=False, debug_title='Pre')
            # likelihood on post data, alpha_one change
            alt_likelihood_alpha_i_post_true = full_likelihood(second_prec_coeffs, H_s, C_two, N=data_two.shape[1], 
                                                            lam=5e-2, include_l1=False, debug_title='Post')
            alt_likelihood_alpha_i_true = alt_likelihood_alpha_i_pre_true + alt_likelihood_alpha_i_post_true
            dof = 2
            test_stat_i_true, p_val_i_true = likelihood_ratio_test(null_likelihood_true, 
                                                alt_likelihood_alpha_i_true, dof, log_pvals=1)
            curr_mat = symmetrize_from_vector(H_s[0], dim)
            nonzero_cols = np.nonzero(np.any(curr_mat != 0, axis=0))[0]
            C_val = len(nonzero_cols)
            if C_val > 0:
                correction_factor = dim/C_val
                #print(correction_factor)
                # correct p_vals for cluster at all time points independently
                p_val_i_true = p_val_i_true*correction_factor
            else:
                p_val_i_true = 1.0
            p_vals_true_curr.append(p_val_i_true)
            lrt_vals_true_curr.append(test_stat_i_true)
            """"""

            """"""
            # est H LRT
            glasso = GraphicalLasso(max_iter=1000, alpha=5e-2, tol=1e-5, verbose=False).fit(data_one.T)
            precision = glasso.precision_.copy()
            clust_dist_mat = np.abs(precision)
            np.fill_diagonal(clust_dist_mat, 0.0)
            ###########
            clust_dist_mat = (clust_dist_mat.max()+1e-5) - clust_dist_mat
            np.fill_diagonal(clust_dist_mat, 0.0)
            ###########
            pairwise_distances = sch.distance.pdist(clust_dist_mat)
            #pairwise_distances = squareform(clust_dist_mat)
            Z = linkage(pairwise_distances, method='complete')
            cutree1 = hierarchy.cut_tree(Z, n_clusters=M).squeeze()
            basis_matrices = []
            for i in range(max(set(cutree1))+1): # iterate over clusters
                idxs = np.where(cutree1 == i)[0] # indexes for given cluster
                A = np.zeros(precision.shape) # blank A matrix
                for idx in idxs: # loop over indexes
                    for idx2 in idxs: # loop over indexes
                        A[idx][idx2] = precision[idx][idx2].copy() # set i,j entry to be the entry from precision matrix for given cluster
                if len(np.nonzero(A)[0]) > 0:
                    basis_matrices.append(vectorize_matrix(A))
            basis_matrices = np.array(basis_matrices)
            assert is_pos_def(symmetrize_from_vector(basis_matrices.sum(axis=0), dim=dim)), "Not PosDef"

            coeffs_hat_total = optim_boyd(C=C_full, H_s=basis_matrices)
            alpha_i_change_pre = coeffs_hat_total.copy()
            alpha_i_change_post = coeffs_hat_total.copy()
            curr_alpha_i_pre = optim_boyd_dc(C=C_one, H=basis_matrices[0])
            curr_alpha_i_post = optim_boyd_dc(C=C_two, H=basis_matrices[0])
            alpha_i_change_pre[0] = curr_alpha_i_pre
            alpha_i_change_post[0] = curr_alpha_i_post
            null_likelihood = full_likelihood(coeffs_hat_total, basis_matrices, C_full, N=data_full.shape[1], 
                                                lam=5e-2, include_l1=False, debug_title='global')
            # # likelihood on pre data, alpha_one change
            alt_likelihood_alpha_i_pre = full_likelihood(alpha_i_change_pre, basis_matrices, C_one, N=data_one.shape[1], 
                                                            lam=5e-2, include_l1=False, debug_title='Pre')
            # likelihood on post data, alpha_one change
            alt_likelihood_alpha_i_post = full_likelihood(alpha_i_change_post, basis_matrices, C_two, N=data_two.shape[1], 
                                                            lam=5e-2, include_l1=False, debug_title='Post')
            alt_likelihood_alpha_i = alt_likelihood_alpha_i_pre + alt_likelihood_alpha_i_post
            dof = 2
            test_stat_i, p_val_i = likelihood_ratio_test(null_likelihood, 
                                                alt_likelihood_alpha_i, dof, log_pvals=1)
            curr_mat = symmetrize_from_vector(basis_matrices[0], dim)
            nonzero_cols = np.nonzero(np.any(curr_mat != 0, axis=0))[0]
            C_val = len(nonzero_cols)
            if C_val > 0:
                correction_factor = dim/C_val
                #print(correction_factor)
                # correct p_vals for cluster at all time points independently
                p_val_i = p_val_i*correction_factor
            else:
                p_val_i = 1.0
            p_vals_curr_est.append(p_val_i)
            lrt_vals_curr_est.append(test_stat_i)
            coeff_diffs_curr_est.append(alpha_i_change_pre[0]-alpha_i_change_post[0])
            """"""

            clime_init = clime_init_fn(5e-2, data_one.T)
            g1 = calc_g1(w)
            g2 = calc_g2(w)
            rhat0 = calc_rhat(clime_init, dim)
            T0 = calc_T_t(X=data_two.T, omega_hat=clime_init, r_hat=rhat0, w=w, p=dim, t=0, g1=g1, g2=g2)
            test_vals_kesh_curr.append(T0)

            clime_init_true = prec_one
            rhat0_true = calc_rhat(clime_init_true, dim)
            T0_true = calc_T_t(X=data_two.T, omega_hat=clime_init_true, r_hat=rhat0_true, w=w, p=dim, t=0, g1=g1, g2=g2)
            test_vals_kesh_true_curr.append(T0_true)
        mag_results['our_lrt'] = np.array(lrt_vals_curr)
        mag_results['our_p_vals'] = np.array(p_vals_curr)
        mag_results['est_diff'] = np.array(coeff_diffs_curr)
        mag_results['kesh_test'] = np.array(test_vals_kesh_curr)
        mag_results['our_lrt_true'] = np.array(lrt_vals_true_curr)
        mag_results['our_p_vals_true'] = np.array(p_vals_true_curr)
        mag_results['kesh_test_true'] = np.array(test_vals_kesh_true_curr)
        mag_results['our_lrt_est'] = np.array(lrt_vals_curr_est)
        mag_results['our_p_vals_est'] = np.array(p_vals_curr_est)
        mag_results['est_diff_est'] = np.array(coeff_diffs_curr_est)
        total_results[curr_change_mag] = mag_results
    
    return total_results

def collect_prec_rec_results(M, w):
    # collect results
    seed_list = np.arange(0, 100)
    coeff_change_mags = np.round(np.arange(0.0, 1.0, 0.1), 2).tolist()
    dim_results = {}
    dims = [20, 40, 60, 80]
    for dim in dims:
        H_s = generate_matrices_orthogonal(M=M, dim=dim, to_print=False)[0]
        prec_one = collect_precision_matrix(H_s=H_s, prec_coeffs=np.ones(M), P=dim)
        total_results = {}
        for curr_change_mag in coeff_change_mags:
            coeff_diffs_curr = []
            coeff_diffs_curr_est = []
            lrt_vals_curr = []
            lrt_vals_true_curr = []
            lrt_vals_curr_est = []
            p_vals_curr = []
            p_vals_true_curr = []
            p_vals_curr_est = []
            test_vals_kesh_curr = []
            test_vals_kesh_true_curr = []
            mag_results = {}
            for curr_seed in tqdm(seed_list):
                second_prec_coeffs = np.ones(M)
                second_prec_coeffs[0] = second_prec_coeffs[0] - curr_change_mag
                prec_two = collect_precision_matrix(H_s=H_s, prec_coeffs=second_prec_coeffs, P=dim)
                data_one, _ = sim_data(covar=inv(prec_one), dim=dim, N=100)
                if dim == 20:
                    data_one_train, _ = sim_data(covar=inv(prec_one), dim=dim, N=100) # identical to data_one, used for fitting the initial Glasso/Clime estimates
                elif dim == 40:
                    data_one_train, _ = sim_data(covar=inv(prec_one), dim=dim, N=200) # identical to data_one, used for fitting the initial Glasso/Clime estimates
                elif dim == 60:
                    data_one_train, _ = sim_data(covar=inv(prec_one), dim=dim, N=300) # identical to data_one, used for fitting the initial Glasso/Clime estimates
                elif dim == 80:
                    data_one_train, _ = sim_data(covar=inv(prec_one), dim=dim, N=400) # identical to data_one, used for fitting the initial Glasso/Clime estimates
                data_two, _ = sim_data(covar=inv(prec_two), dim=dim, N=100)
                data_full = np.concatenate((data_one, data_two), axis=1)
                data_full_train = np.concatenate((data_one_train, data_two), axis=1)
                C_one = np.cov(data_one, bias=True)
                C_one_train = np.cov(data_one_train, bias=True)
                C_two = np.cov(data_two, bias=True)
                C_full = np.cov(data_full, bias=True)
                C_full_train = np.cov(data_full_train, bias=True)
                coeffs_hat_total = optim_boyd(C=C_full, H_s=H_s)
                alpha_i_change_pre = coeffs_hat_total.copy()
                alpha_i_change_post = coeffs_hat_total.copy()
                curr_alpha_i_pre = optim_boyd_dc(C=C_one, H=H_s[0])
                curr_alpha_i_post = optim_boyd_dc(C=C_two, H=H_s[0])
                alpha_i_change_pre[0] = curr_alpha_i_pre
                alpha_i_change_post[0] = curr_alpha_i_post
                null_likelihood = full_likelihood(coeffs_hat_total, H_s, C_full, N=data_full.shape[1], 
                                                    lam=5e-2, include_l1=False, debug_title='global')
                # # likelihood on pre data, alpha_one change
                alt_likelihood_alpha_i_pre = full_likelihood(alpha_i_change_pre, H_s, C_one, N=data_one.shape[1], 
                                                                lam=5e-2, include_l1=False, debug_title='Pre')
                # likelihood on post data, alpha_one change
                alt_likelihood_alpha_i_post = full_likelihood(alpha_i_change_post, H_s, C_two, N=data_two.shape[1], 
                                                                lam=5e-2, include_l1=False, debug_title='Post')
                alt_likelihood_alpha_i = alt_likelihood_alpha_i_pre + alt_likelihood_alpha_i_post
                dof = 2
                test_stat_i, p_val_i = likelihood_ratio_test(null_likelihood, 
                                                    alt_likelihood_alpha_i, dof, log_pvals=1)
                curr_mat = symmetrize_from_vector(H_s[0], dim)
                nonzero_cols = np.nonzero(np.any(curr_mat != 0, axis=0))[0]
                C_val = len(nonzero_cols)
                if C_val > 0:
                    correction_factor = dim/C_val
                    #print(correction_factor)
                    # correct p_vals for cluster at all time points independently
                    p_val_i = p_val_i*correction_factor
                else:
                    p_val_i = 1.0
                p_vals_curr.append(p_val_i)
                lrt_vals_curr.append(test_stat_i)
                coeff_diffs_curr.append(alpha_i_change_pre[0]-alpha_i_change_post[0])

                """"""
                # true LRT
                null_likelihood_true = full_likelihood(coeffs_hat_total, H_s, C_full, N=data_full.shape[1], 
                                                    lam=5e-2, include_l1=False, debug_title='global')
                # # likelihood on pre data, alpha_one change
                alt_likelihood_alpha_i_pre_true = full_likelihood(np.ones(M), H_s, C_one, N=data_one.shape[1], 
                                                                lam=5e-2, include_l1=False, debug_title='Pre')
                # likelihood on post data, alpha_one change
                alt_likelihood_alpha_i_post_true = full_likelihood(second_prec_coeffs, H_s, C_two, N=data_two.shape[1], 
                                                                lam=5e-2, include_l1=False, debug_title='Post')
                alt_likelihood_alpha_i_true = alt_likelihood_alpha_i_pre_true + alt_likelihood_alpha_i_post_true
                dof = 2
                test_stat_i_true, p_val_i_true = likelihood_ratio_test(null_likelihood_true, 
                                                    alt_likelihood_alpha_i_true, dof, log_pvals=1)
                curr_mat = symmetrize_from_vector(H_s[0], dim)
                nonzero_cols = np.nonzero(np.any(curr_mat != 0, axis=0))[0]
                C_val = len(nonzero_cols)
                if C_val > 0:
                    correction_factor = dim/C_val
                    #print(correction_factor)
                    # correct p_vals for cluster at all time points independently
                    p_val_i_true = p_val_i_true*correction_factor
                else:
                    p_val_i_true = 1.0
                p_vals_true_curr.append(p_val_i_true)
                lrt_vals_true_curr.append(test_stat_i_true)
                """"""

                """"""
                # est H LRT
                #glasso = GraphicalLasso(max_iter=1000, alpha=5e-2, tol=1e-5, verbose=False).fit(data_one.T)
                glasso = GraphicalLasso(max_iter=1000, alpha=5e-2, tol=1e-5, verbose=False).fit(data_one_train.T) # this one has more data used
                precision = glasso.precision_.copy()
                clust_dist_mat = np.abs(precision)
                np.fill_diagonal(clust_dist_mat, 0.0)
                ###########
                clust_dist_mat = (clust_dist_mat.max()+1e-5) - clust_dist_mat
                np.fill_diagonal(clust_dist_mat, 0.0)
                ###########
                pairwise_distances = sch.distance.pdist(clust_dist_mat)
                #pairwise_distances = squareform(clust_dist_mat)
                Z = linkage(pairwise_distances, method='complete')
                cutree1 = hierarchy.cut_tree(Z, n_clusters=M).squeeze()
                basis_matrices = []
                for i in range(max(set(cutree1))+1): # iterate over clusters
                    idxs = np.where(cutree1 == i)[0] # indexes for given cluster
                    A = np.zeros(precision.shape) # blank A matrix
                    for idx in idxs: # loop over indexes
                        for idx2 in idxs: # loop over indexes
                            A[idx][idx2] = precision[idx][idx2].copy() # set i,j entry to be the entry from precision matrix for given cluster
                    if len(np.nonzero(A)[0]) > 0:
                        basis_matrices.append(vectorize_matrix(A))
                basis_matrices = np.array(basis_matrices)
                curr_silhoutte_score = silhouette_score(clust_dist_mat, cutree1, metric='precomputed')

                basis_matrices = bfs_basis_mats(data_one_train, basis_matrices, curr_silhoutte_score, dim=dim, recursion_min=2, cutree=cutree1, root_dist_mat=clust_dist_mat)
                assert is_pos_def(symmetrize_from_vector(basis_matrices.sum(axis=0), dim=dim)), "Not PosDef"
                
                coeffs_hat_total = optim_boyd(C=C_full, H_s=basis_matrices)
                coeffs_hat_total_train = optim_boyd(C=C_full_train, H_s=basis_matrices)
                p_val_min = np.inf
                test_stat_max = 0
                for i in range(M):
                    alpha_i_change_pre = coeffs_hat_total_train.copy()
                    alpha_i_change_post = coeffs_hat_total_train.copy()
                    curr_alpha_i_pre = optim_boyd_dc(C=C_one_train, H=basis_matrices[i])
                    curr_alpha_i_post = optim_boyd_dc(C=C_two, H=basis_matrices[i])
                    alpha_i_change_pre[i] = curr_alpha_i_pre
                    alpha_i_change_post[i] = curr_alpha_i_post
                    null_likelihood = full_likelihood(coeffs_hat_total_train, basis_matrices, C_full_train, N=data_full_train.shape[1], 
                                                        lam=5e-2, include_l1=False, debug_title='global')
                    # # likelihood on pre data, alpha_one change
                    alt_likelihood_alpha_i_pre = full_likelihood(alpha_i_change_pre, basis_matrices, C_one_train, N=data_one_train.shape[1], 
                                                                    lam=5e-2, include_l1=False, debug_title='Pre')
                    # likelihood on post data, alpha_one change
                    alt_likelihood_alpha_i_post = full_likelihood(alpha_i_change_post, basis_matrices, C_two, N=data_two.shape[1], 
                                                                    lam=5e-2, include_l1=False, debug_title='Post')
                    alt_likelihood_alpha_i = alt_likelihood_alpha_i_pre + alt_likelihood_alpha_i_post
                    dof = 2
                    test_stat_i, p_val_i = likelihood_ratio_test(null_likelihood, 
                                                        alt_likelihood_alpha_i, dof, log_pvals=1)
                    curr_mat = symmetrize_from_vector(basis_matrices[i], dim)
                    nonzero_cols = np.nonzero(np.any(curr_mat != 0, axis=0))[0]
                    C_val = len(nonzero_cols)
                    if C_val > 0:
                        correction_factor = dim/C_val
                        #print(correction_factor)
                        # correct p_vals for cluster at all time points independently
                        p_val_i = p_val_i*correction_factor
                    else:
                        p_val_i = 1.0
                    if p_val_i < p_val_min:
                        p_val_min = p_val_i
                        test_stat_max = test_stat_i
                p_vals_curr_est.append(p_val_min)
                lrt_vals_curr_est.append(test_stat_max)
                coeff_diffs_curr_est.append(alpha_i_change_pre[0]-alpha_i_change_post[0]) # this shouldn't be used for anything really



                """"""

                #clime_init = clime_init_fn(5e-2, data_one.T)
                clime_init = clime_init_fn(5e-2, data_one_train.T) # this one has more data used
                g1 = calc_g1(w)
                g2 = calc_g2(w)
                rhat0 = calc_rhat(clime_init, dim)
                T0 = calc_T_t(X=data_two.T, omega_hat=clime_init, r_hat=rhat0, w=w, p=dim, t=0, g1=g1, g2=g2)
                test_vals_kesh_curr.append(T0)

                clime_init_true = prec_one
                rhat0_true = calc_rhat(clime_init_true, dim)
                T0_true = calc_T_t(X=data_two.T, omega_hat=clime_init_true, r_hat=rhat0_true, w=w, p=dim, t=0, g1=g1, g2=g2)
                test_vals_kesh_true_curr.append(T0_true)
            mag_results['our_lrt'] = np.array(lrt_vals_curr)
            mag_results['our_p_vals'] = np.array(p_vals_curr)
            mag_results['est_diff'] = np.array(coeff_diffs_curr)
            mag_results['kesh_test'] = np.array(test_vals_kesh_curr)
            mag_results['our_lrt_true'] = np.array(lrt_vals_true_curr)
            mag_results['our_p_vals_true'] = np.array(p_vals_true_curr)
            mag_results['kesh_test_true'] = np.array(test_vals_kesh_true_curr)
            mag_results['our_lrt_est'] = np.array(lrt_vals_curr_est)
            mag_results['our_p_vals_est'] = np.array(p_vals_curr_est)
            mag_results['est_diff_est'] = np.array(coeff_diffs_curr_est)
            total_results[str(curr_change_mag)] = mag_results
            #print(mag_results['our_p_vals'].shape, mag_results['our_p_vals_true'].shape, mag_results['our_p_vals_est'].shape)
        dim_results[dim] = total_results
    
    with open('lrt_test_figs/dim_results.pkl', 'wb') as fp:
        pickle.dump(dim_results, fp)

    return dim_results

def process_curr_pr(change_vals, no_change_vals, pvals=True):
    all_thresholds = np.sort(np.concatenate((change_vals, no_change_vals)))
    precisions = []
    recalls = []
    fprates = []
    for thr in all_thresholds:
        if pvals:
            exceed_mask_change = (change_vals <= thr)
            exceed_mask_no_change = (no_change_vals <= thr)
        else:
            exceed_mask_change = (change_vals >= thr)
            exceed_mask_no_change = (no_change_vals >= thr)
        true_positives = np.sum(exceed_mask_change)
        false_positives = np.sum(exceed_mask_no_change)
        true_negatives = len(no_change_vals) - false_positives
        false_negatives = len(change_vals) - true_positives

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives) # also TPR
        fprate = false_positives / (false_positives + true_negatives)
        precisions.append(precision)
        recalls.append(recall)
        fprates.append(fprate)
    
    return np.array(precisions), np.array(recalls), np.array(fprates)

def plot_precision_recall(prec, rec, dim, labl='Ours'):
    plt.plot(rec, prec)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.title("Precision Recall Dim {} {}".format(dim, labl))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.tight_layout()
    plt.savefig('lrt_test_figs/prec_rec/{}'.format("prec_rec_{}_{}.png".format(dim, labl)))
    plt.close()

def plot_precision_recall_comparison(prec_ours, rec_ours, prec_other, rec_other, dim, labl='Ours'):
    plt.plot(rec_ours, prec_ours, label='Ours')
    plt.plot(rec_other, prec_other, label='Kesh')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.title("Precision Recall Dim {} {}".format(dim, labl))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.tight_layout()
    plt.legend(loc='best')
    plt.savefig('lrt_test_figs/prec_rec_compare/{}'.format("prec_rec_{}_{}.png".format(dim, labl)))
    plt.close()

def plot_roc(tprate, fprate, dim, labl='Ours'):
    plt.plot(fprate, tprate)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.title("ROC Dim {} {}".format(dim, labl))
    plt.xlabel('FPRate')
    plt.ylabel('TPRate')
    plt.tight_layout()
    plt.savefig('lrt_test_figs/roc/{}'.format("roc_{}_{}.png".format(dim, labl)))
    plt.close()

def plot_roc_comparison(tprate_ours, fprate_ours, tprate_other, fprate_other, dim, labl='Ours'):
    plt.plot(fprate_ours, tprate_ours, label='Ours')
    plt.plot(fprate_other, tprate_other, label='Kesh')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.title("ROC Dim {} {}".format(dim, labl))
    plt.xlabel('FPRate')
    plt.ylabel('TPRate')
    plt.tight_layout()
    plt.legend(loc='best')
    plt.savefig('lrt_test_figs/roc_compare/{}'.format("roc_{}_{}.png".format(dim, labl)))
    plt.close()

def process_prec_rec_results(dim_results):
    print("Plotting Results...")
    seed_list = np.arange(0, 100)
    coeff_change_mags = np.round(np.arange(0.1, 1.0, 0.1), 2).tolist() # just iterate over the actual changes, compare vs no change
    dims = [20, 40, 60, 80]
    for dim in dims:
        curr_dim_no_change_ours = dim_results[dim][0.0]['our_p_vals'] # with H given
        curr_dim_no_change_ours_est = dim_results[dim][0.0]['our_p_vals_est'] # nothing given
        curr_dim_no_change_ours_true = dim_results[dim][0.0]['our_p_vals_true'] # everything given

        curr_dim_no_change_kesh = dim_results[dim][0.0]['kesh_test'] # with nothing given
        curr_dim_no_change_kesh_true = dim_results[dim][0.0]['kesh_test_true'] # everything given
        for mag in tqdm(coeff_change_mags):
            curr_dim_change_ours = dim_results[dim][mag]['our_p_vals'] # with H given
            curr_dim_change_ours_est = dim_results[dim][mag]['our_p_vals_est'] # nothing given
            curr_dim_change_ours_true = dim_results[dim][mag]['our_p_vals_true'] # everything given

            curr_dim_change_kesh = dim_results[dim][mag]['kesh_test'] # with nothing given
            curr_dim_change_kesh_true = dim_results[dim][mag]['kesh_test_true'] # everything given

            prec_ours, rec_ours, fprate_ours = process_curr_pr(change_vals=curr_dim_change_ours, no_change_vals=curr_dim_no_change_ours)
            prec_ours_est, rec_ours_est, fprate_ours_est = process_curr_pr(change_vals=curr_dim_change_ours_est, no_change_vals=curr_dim_no_change_ours_est)
            prec_ours_true, rec_ours_true, fprate_ours_true = process_curr_pr(change_vals=curr_dim_change_ours_true, no_change_vals=curr_dim_no_change_ours_true)

            prec_kesh, rec_kesh, fprate_kesh = process_curr_pr(change_vals=curr_dim_change_kesh, no_change_vals=curr_dim_no_change_kesh, pvals=False)
            prec_kesh_true, rec_kesh_true, fprate_kesh_true = process_curr_pr(change_vals=curr_dim_change_kesh_true, no_change_vals=curr_dim_no_change_kesh_true, pvals=False)

            plot_precision_recall(prec_ours, rec_ours, dim=dim, labl='Ours_Magnitude_{}'.format(mag))
            plot_precision_recall(prec_ours_est, rec_ours_est, dim=dim, labl='Ours_Est_Magnitude_{}'.format(mag))
            plot_precision_recall(prec_ours_true, rec_ours_true, dim=dim, labl='Ours_True_Magnitude_{}'.format(mag))

            plot_precision_recall(prec_kesh, rec_kesh, dim=dim, labl='Kesh_Magnitude_{}'.format(mag))
            plot_precision_recall(prec_kesh_true, rec_kesh_true, dim=dim, labl='Kesh_True_Magnitude_{}'.format(mag))

            plot_roc(rec_ours, fprate_ours, dim=dim, labl='Ours_Magnitude_{}'.format(mag))
            plot_roc(rec_ours_est, fprate_ours_est, dim=dim, labl='Ours_Est_Magnitude_{}'.format(mag))
            plot_roc(rec_ours_true, fprate_ours_true, dim=dim, labl='Ours_True_Magnitude_{}'.format(mag))

            plot_roc(rec_kesh, fprate_kesh, dim=dim, labl='Kesh_Magnitude_{}'.format(mag))
            plot_roc(rec_kesh_true, fprate_kesh_true, dim=dim, labl='Kesh_True_Magnitude_{}'.format(mag))

            # comparisons, the important part
            plot_precision_recall_comparison(prec_ours, rec_ours, prec_kesh, rec_kesh, dim=dim, labl='Ours_Kesh_Magnitude_{}'.format(mag))
            plot_roc_comparison(tprate_ours=rec_ours, fprate_ours=fprate_ours, tprate_other=rec_kesh, fprate_other=fprate_kesh, dim=dim, labl='Ours_Kesh_Magnitude_{}'.format(mag))

            plot_precision_recall_comparison(prec_ours_est, rec_ours_est, prec_kesh, rec_kesh, dim=dim, labl='Ours_Estimate_Kesh_Magnitude_{}'.format(mag))
            plot_roc_comparison(tprate_ours=rec_ours_est, fprate_ours=fprate_ours_est, tprate_other=rec_kesh, fprate_other=fprate_kesh, dim=dim, labl='Ours_Estimate_Kesh_Magnitude_{}'.format(mag))

            plot_precision_recall_comparison(prec_ours_true, rec_ours_true, prec_kesh_true, rec_kesh_true, dim=dim, labl='Ours_True_Kesh_Magnitude_{}'.format(mag))
            plot_roc_comparison(tprate_ours=rec_ours_true, fprate_ours=fprate_ours_true, tprate_other=rec_kesh_true, fprate_other=fprate_kesh_true, dim=dim, labl='Ours_True_Kesh_Magnitude_{}'.format(mag))
    

def plot_results(res):
    coeff_change_mags = np.arange(0.0, 1.0, 0.1)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"])
    plt.xticks(rotation=45, ha='right')
    ylim_max = 10.0
    for i in range(1, 11):
        plot_confidence_interval(x=i, values=res[coeff_change_mags[i-1]]['est_diff'], z=1.96, color='#2187bb', horizontal_line_width=0.25)
    #plt.ylim(0, ylim_max)
    plt.title("Confidence Intervals {}".format("Differences"))
    plt.ylabel("Estimated Difference")
    plt.xlabel("Coefficient Difference")
    plt.tight_layout()
    plt.savefig('lrt_test_figs/{}'.format("conf_int_diff_total.png"))
    plt.close()

    coeff_change_mags = np.arange(0.0, 1.0, 0.1)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"])
    plt.xticks(rotation=45, ha='right')
    ylim_max = 10.0
    for i in range(1, 11):
        plot_confidence_interval(x=i, values=res[coeff_change_mags[i-1]]['est_diff_est'], z=1.96, color='#2187bb', horizontal_line_width=0.25)
    #plt.ylim(0, ylim_max)
    plt.title("Confidence Intervals {}".format("Differences"))
    plt.ylabel("Estimated Difference")
    plt.xlabel("Coefficient Difference")
    plt.tight_layout()
    plt.savefig('lrt_test_figs/{}'.format("conf_int_diff_total_est.png"))
    plt.close()

    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"])
    plt.xticks(rotation=45, ha='right')
    for i in range(1, 11):
        plot_confidence_interval(x=i, values=res[coeff_change_mags[i-1]]['our_p_vals'], z=1.96, color='#2187bb', horizontal_line_width=0.25)
    #plt.ylim(0, ylim_max)
    plt.title("Confidence Intervals {}".format("log(P Values)"))
    plt.ylabel("log(P Value)")
    plt.xlabel("Coefficient Difference")
    plt.tight_layout()
    plt.savefig('lrt_test_figs/{}'.format("conf_int_pvals_total.png"))
    plt.close()
    
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"])
    plt.xticks(rotation=45, ha='right')
    for i in range(1, 11):
        plot_confidence_interval(x=i, values=res[coeff_change_mags[i-1]]['our_p_vals_est'], z=1.96, color='#2187bb', horizontal_line_width=0.25)
    #plt.ylim(0, ylim_max)
    plt.title("Confidence Intervals {}".format("log(P Values)"))
    plt.ylabel("log(P Value)")
    plt.xlabel("Coefficient Difference")
    plt.tight_layout()
    plt.savefig('lrt_test_figs/{}'.format("conf_int_pvals_total_est.png"))
    plt.close()

    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"])
    plt.xticks(rotation=45, ha='right')
    for i in range(1, 11):
        plot_confidence_interval(x=i, values=res[coeff_change_mags[i-1]]['our_p_vals_true'], z=1.96, color='#2187bb', horizontal_line_width=0.25)
    #plt.ylim(0, ylim_max)
    plt.title("Confidence Intervals True {}".format("log(P Values)"))
    plt.ylabel("log(P Value)")
    plt.xlabel("Coefficient Difference")
    plt.tight_layout()
    plt.savefig('lrt_test_figs/{}'.format("conf_int_pvals_true_total.png"))
    plt.close()

    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"])
    plt.xticks(rotation=45, ha='right')
    for i in range(1, 11):
        plot_confidence_interval(x=i, values=res[coeff_change_mags[i-1]]["kesh_test"], z=1.96, color='#2187bb', horizontal_line_width=0.25)
    plt.title("Confidence Intervals {}".format("Test Stat Kesh"))
    plt.ylabel("Test Stat")
    plt.xlabel("Coefficient Difference")
    plt.tight_layout()
    plt.savefig('lrt_test_figs/{}'.format("conf_int_tstats_kesh_total.png"))
    plt.close()

    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"])
    plt.xticks(rotation=45, ha='right')
    for i in range(1, 11):
        plot_confidence_interval(x=i, values=res[coeff_change_mags[i-1]]["kesh_test_true"], z=1.96, color='#2187bb', horizontal_line_width=0.25)
    plt.title("Confidence Intervals True {}".format("Test Stat Kesh"))
    plt.ylabel("Test Stat")
    plt.xlabel("Coefficient Difference")
    plt.tight_layout()
    plt.savefig('lrt_test_figs/{}'.format("conf_int_tstats_kesh_true_total.png"))
    plt.close()


def main(args):
    if args.prec_rec:
        print("Collecting Prec/Rec results")
        if args.load_results:
            with open('lrt_test_figs/dim_results.pkl', 'rb') as fp:
                res = pickle.load(fp)
                dims = [20, 40, 60, 80]
                for dim in dims:
                    import copy
                    currkeys = [float(x) for x in res[dim].keys()]
                    print(currkeys)
                    for k in currkeys:
                        vals = res[dim].pop(str(k))
                        res[dim][round(k, 2)] = vals
                print("> Results Loaded! <")            
        else:
            res = collect_prec_rec_results(M=args.M, w=args.w)
        prec_rec_res = process_prec_rec_results(dim_results=res)
    else:
        res = collect_test_results(M=args.M, dim=args.dim, w=args.w)
        plot_results(res)
    

    


if __name__ == '__main__':
    args = get_args()
    main(args)