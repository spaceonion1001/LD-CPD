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

from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2, multivariate_normal
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, silhouette_score, pairwise_distances as pairwise_d
from sklearn.utils.extmath import fast_logdet
from sklearn.preprocessing import StandardScaler

from inverse_covariance import QuicGraphicalLasso

from numba import jit


from tqdm import tqdm
from simulate import generate_matrices_orthogonal, sim_data, collect_precision_matrix
from likelihood import full_likelihood, likelihood_ratio_test
from numpy.linalg import inv as inv
from optim import optim_boyd, optim_boyd_dc
from utils import symmetrize_from_vector, vectorize_matrix, is_pos_def, scale_data
from meinshausen import meinshausen_correction
np.random.seed(42)
import argparse

import warnings
warnings.filterwarnings('ignore')  # <- remember to comment this if something breaks and you get confused

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
from thav_gl import thav_gl_fn

from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as sch
import scipy

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lam', type=float, default=1e-1)
    parser.add_argument('--M', type=int, default=8)
    parser.add_argument('--dim', type=int, default=80)
    parser.add_argument('--w', type=int, default=100)
    parser.add_argument('--prec_rec', type=int, default=0)
    parser.add_argument('--load_results', type=int, default=0)
    parser.add_argument('--linkage', type=str, default='single')
    parser.add_argument('--dims', type=str, default='20,40,60,80')
    args = parser.parse_args()

    return args

def plot_confidence_interval(x, values, z=1.96, color='#2187bb', horizontal_line_width=0.25, constrain=False):
    mean = statistics.mean(values)
    #median = statistics.median(values)
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
    #plt.plot(x, median, 'o', color='yellow')

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

def perform_regression(X_data, Y_data, kd=2):
    """
    Split data in half and perform the regression process on each column
    """
    #half = data.shape[0]//2
    #X_data = data[:half, :]
    #Y_data = data[half:, :]
    dim = X_data.shape[1]
    x_0 = np.random.normal(0, 1, dim-1)
    x_log_dim = np.log(dim)/X_data.shape[0]
    y_log_dim = np.log(dim)/Y_data.shape[0]
    beta_hats_x = []
    beta_hats_y = []
    residuals_x = np.zeros((X_data.shape[0], dim))
    residuals_y = np.zeros((Y_data.shape[0], dim))
    beta_hats_x = np.zeros((dim, dim-1))
    beta_hats_y = np.zeros((dim, dim-1))
    for i in range(dim):
        X_m_i = np.delete(X_data, i, axis=1)
        y_i = X_data[:, i]
        sig_i_i_x = np.var(y_i)
        x_lam = kd*np.power(sig_i_i_x*x_log_dim, 0.5)
        nrow, ncol = X_m_i.shape
        X = r.matrix(X_m_i, nrow=nrow, ncol=ncol)
        y_i = robjects.FloatVector(X_data[:, i])
        reg_soln = scalreg.scalreg(X, y_i, lam0=x_lam)
        reg_soln_dict = dict(zip(reg_soln.names, list(reg_soln)))
        residuals_x[:, i] = np.array(reg_soln_dict['residuals'])
        beta_hats_x_curr = np.array(reg_soln_dict['coefficients'])
        beta_hats_x[i, :] = beta_hats_x_curr

        Y_m_i = np.delete(Y_data, i, axis=1)
        yy_i = Y_data[:, i]
        sig_i_i_y = np.var(yy_i)
        y_lam = kd*np.power(sig_i_i_y*y_log_dim, 0.5)
        nrow, ncol = Y_m_i.shape
        Y = r.matrix(Y_m_i, nrow=nrow, ncol=ncol)
        yy_i = robjects.FloatVector(Y_data[:, i])
        reg_soln = scalreg.scalreg(Y, yy_i, lam0=y_lam)
        reg_soln_dict = dict(zip(reg_soln.names, list(reg_soln)))
        residuals_y[:, i] = np.array(reg_soln_dict['residuals'])
        beta_hats_y_curr = np.array(reg_soln_dict['coefficients'])
        beta_hats_y[i, :] = beta_hats_y_curr

    return residuals_x, residuals_y, beta_hats_x, beta_hats_y

@jit(nopython=True)
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


@jit(nopython=True)
def calculate_T(residuals_cov_corrected):
    dim = residuals_cov_corrected.shape[0]
    T = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            top = residuals_cov_corrected[i, j]
            bottom = residuals_cov_corrected[i,i]*residuals_cov_corrected[j,j]
            T[i, j] = top/bottom
    
    return T

@jit(nopython=True)
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

@jit(nopython=True)
def calculate_standardized_stat(T_x, T_y, theta_x, theta_y):
    W = (T_x - T_y)/np.sqrt(theta_x+theta_y)

    return W

@jit(nopython=True)
def calculate_global_stat(W):
    M = np.max(W**2)

    return M

# def recurse_on_candidate_point(self, lrt_vals_all, p_vals_corrected, data_full, basis_mats, candidate_point):
#     """
#     Input is p-vals from current level of recursion AT candidate point
#     Shape: (1, M)

#     This function recurses down a level
#     """
#     greatest_change_mat_idx = p_vals_corrected[0, :].argmin() # this should work if it's 1-d just expanded
#     greatest_change_mat = symmetrize_from_vector(basis_mats[greatest_change_mat_idx], dim=self.dim)
#     nonzero_cols = np.nonzero(np.any(greatest_change_mat != 0, axis=0))[0]
#     #print(data_full.shape)
#     print("GREATEST CHANGE MATRIX CHANNELS CONTAINED {}".format(nonzero_cols))
#     #print(p_vals_corrected.shape)

#     # store data for recursion - cp identification
#     data_one = data_full[:, 0:self.args.window_size]
#     data_two = data_full[:, candidate_point:(candidate_point + self.post_window_size)]
#     data_total_window = np.concatenate((data_one, data_two), axis=1)
#     C_one = np.cov(data_one, bias=True)
#     C_one = C_one + np.eye(C_one.shape[0])*1e-8
#     C_two = np.cov(data_two, bias=True)
#     C_two = C_two + np.eye(C_two.shape[0])*1e-8 # correct numerical instability
#     C_total = np.cov(data_total_window, bias=True)
#     C_total = C_total + np.eye(C_total.shape[0])*1e-8
    
#     # store training data for model fitting - goodness of fit check
#     data_train = data_full[:, 0:int(self.args.train_percent*data_full.shape[1])]
#     #print("TRAIN DATA {}".format(data_train.shape))
#     C_full = np.cov(data_train.copy(), bias=True)
#     data_train = data_train[nonzero_cols, :]
#     if len(nonzero_cols) < self.args.recursion_min: # rerun it and stop
#         lrt_vals_all, p_vals_all = LRT_individual_coeffs_full_likelihood(data_full, M=basis_mats.shape[0], dim=data_full.shape[0], H_s=basis_mats, 
#                                                                         window_size=self.window_size, post_window_size=self.post_window_size, lam=self.lam, step_size=self.step_size, include_l1=self.include_l1, 
#                                                                         iters=self.iters, beta=self.beta, t=self.t, optim_type=self.optim_type, args=self.args)
        
#         lrt_vals_all = np.array(lrt_vals_all)
#         p_vals_all = np.array(p_vals_all)
#         #p_vals_corrected = np.array(apply_bonferroni_correction(p_vals_all))
#         #p_vals_corrected = np.array(apply_fdr_correction(p_vals_all))
#         p_vals_corrected = meinshausen_correction(basis_mats, p_vals_all, dim=data_full.shape[0], log_pvals=self.args.log_pvals)

#         return lrt_vals_all, p_vals_corrected
#     train_C = np.cov(data_train.copy(), bias=True)
#     train_C = train_C + np.eye(train_C.shape[0])*1e-8

#     # fit current level of recursion on training data
#     alphas = optim_boyd(C=C_full, H_s=basis_mats)
#     #print("ALPHAS BOYD {}".format(alphas))
#     #print("Train C Shape {}".format(train_C.shape))
#     #print("Alphas {}".format(alphas))
#     cluster_precision = greatest_change_mat[~np.all(greatest_change_mat == 0, axis=1)]
#     cluster_precision = alphas[greatest_change_mat_idx]*cluster_precision[:, ~np.all(cluster_precision == 0, axis=0)]

#     anderson_lrt_value = self.anderson_lrt(cluster_precision=cluster_precision, C=train_C, N=data_train.shape[1])
#     dof = 0.5*train_C.shape[0]*(train_C.shape[0]+1) - 1 # q here is just 1 since we are cluster specific
#     chisquare_val = chi2.sf(anderson_lrt_value, dof)

#     cluster_eigvals = sorted(np.linalg.eig(cluster_precision)[0], reverse=True)
#     cluster_eigval_max = max(cluster_eigvals)
#     cluster_eigval_min = min(cluster_eigvals)
#     eigval_ratio = cluster_eigval_max/cluster_eigval_min # condition number
#     print("Eigvals", cluster_eigvals)
#     print("Eigval ratio", eigval_ratio)
#     #print("CHISQUARE P-VAL {} DOF {}".format(chisquare_val, dof))
#     # if conditions are met, recurse
#     if len(nonzero_cols) >= self.args.recursion_min and eigval_ratio >= self.args.condition_number_thresh: # first stopping conditions
#         new_basis_matrices = self.recursive_split_basis_matrix(basis_mats, greatest_change_mat_idx)
#         #print("NUMBER OF NEW BASIS MATRICES {}".format(new_basis_matrices.shape[0]))
#         nonzero_cols_one = np.nonzero(np.any(symmetrize_from_vector(new_basis_matrices[0], self.dim) != 0, axis=0))[0]
#         if new_basis_matrices.shape[0] > 1:
#             #print("RECALCULATING SILHOUETTE SCORE")
#             #print("CURR CUTREE", self.cutree)
#             nonzero_cols_two = np.nonzero(np.any(symmetrize_from_vector(new_basis_matrices[1], self.dim) != 0, axis=0))[0]
#             #print("NONZERO COLS", nonzero_cols_two)
#             if len(nonzero_cols_one) >= self.args.recursion_min or len(nonzero_cols_two) >= self.args.recursion_min: # if the clustering is able to be split, recurse
#                 self.cutree[nonzero_cols_two] = int(self.cutree.max() + 1)
#                 #print("NEW CUTREE", self.cutree)
#                 new_silhoutte_score = silhouette_score(self.root_dist_mat, self.cutree, metric='precomputed')
#                 #print("Silhoutte Scores", new_silhoutte_score, self.curr_silhoutte_score)
#                 reduced_basis_mats = np.delete(basis_mats, greatest_change_mat_idx, axis=0)
#                 updated_basis_matrices = np.concatenate((reduced_basis_mats, new_basis_matrices), axis=0)
#                 self.basis_matrices = updated_basis_matrices
#                 condition_one = new_basis_matrices.shape[0] > 1
#                 condition_two = new_silhoutte_score > self.curr_silhoutte_score
#                 condition_three = chisquare_val >= 1e-5
#                 #print("Condition One {} Two {} Three {}".format(condition_one, condition_two, condition_three))
#                 #print("\n\n############# RECURSING #################\n")
#                 self.curr_silhoutte_score = new_silhoutte_score
#                 _, p_vals_corrected = self.fit_optim_candidate_point(C_one=C_one, C_two=C_two, C_full=C_total, H_s=basis_mats, window_size=self.window_size, lam=self.lam)
#                 return self.recurse_on_candidate_point(lrt_vals_all, p_vals_corrected=p_vals_corrected, data_full=data_full, basis_mats=self.basis_matrices, candidate_point=candidate_point)
#             else:
#                 print("\n\n************** NOT RECURSING - TOO SMALL OF SPLIT {} {}**************\n".format(len(nonzero_cols_one), len(nonzero_cols_two)))
#         else:
#             print("\n\n************** NOT RECURSING - BASIS MATRIX SHAPE **************\n")


#     lrt_vals_all, p_vals_all = LRT_individual_coeffs_full_likelihood(data_full, M=basis_mats.shape[0], dim=data_full.shape[0], H_s=basis_mats, 
#                                                                         window_size=self.window_size, post_window_size=self.post_window_size, lam=self.lam, step_size=self.step_size, include_l1=self.include_l1, 
#                                                                         iters=self.iters, beta=self.beta, t=self.t, optim_type=self.optim_type, args=self.args)
    
#     lrt_vals_all = np.array(lrt_vals_all)
#     p_vals_all = np.array(p_vals_all)
#     #p_vals_corrected = np.array(apply_bonferroni_correction(p_vals_all))
#     #p_vals_corrected = np.array(apply_fdr_correction(p_vals_all))
#     p_vals_corrected = meinshausen_correction(basis_mats, p_vals_all, dim=data_full.shape[0], log_pvals=self.args.log_pvals)

#     return lrt_vals_all, p_vals_corrected


def recursive_split_basis_matrix(basis_mats, greatest_change_mat_idx, dim):
    greatest_change_mat = symmetrize_from_vector(basis_mats[greatest_change_mat_idx], dim=dim)
    orig_nonzero_cols = np.nonzero(np.any(greatest_change_mat != 0, axis=0))[0]
    basis_mat_reduced = greatest_change_mat[:, ~np.all(greatest_change_mat == 0, axis=0)]
    basis_mat_reduced = basis_mat_reduced[~np.all(basis_mat_reduced == 0, axis=1), :]
    #print("GREATEST CHANGE IDX: ", greatest_change_mat_idx, greatest_change_mat.shape)
    # RECLUSTER - SIMPLEST SOLUTION CURRENTLY
    #clust_dist_mat = np.abs(greatest_change_mat)
    clust_dist_mat = np.abs(basis_mat_reduced)
    np.fill_diagonal(clust_dist_mat, 0.0)
    ###########
    clust_dist_mat = (clust_dist_mat.max()+1e-5) - clust_dist_mat
    np.fill_diagonal(clust_dist_mat, 0.0)
    ###########
    #pairwise_distances = sch.distance.pdist(clust_dist_mat)
    pairwise_distances = squareform(clust_dist_mat)
    Z = linkage(pairwise_distances, method='single')
    # BREAK INTO 2 NEW CLUSTERS
    cutree1 = hierarchy.cut_tree(Z, n_clusters=2).squeeze()
    #fclust_res = fcluster(Z, t=2, criterion='maxclust')
    ###################
    #cutree = fclust_res
    # new_silhoutte_score = silhouette_score(clust_dist_mat, cutree, metric='precomputed')
    ###################
    new_basis_matrices = []
    for i in range(min(set(cutree1)), max(set(cutree1))+1): # iterate over clusters
        idxs = np.where(cutree1 == i)[0] # indexes for given cluster
        A = np.zeros(greatest_change_mat.shape) # blank A matrix
        for idx in idxs: # loop over indexes
            for idx2 in idxs: # loop over indexes
                first_idx = orig_nonzero_cols[idx] # remap back to original space
                second_idx = orig_nonzero_cols[idx2] # remap back to original space
                A[first_idx][second_idx] = greatest_change_mat[first_idx][second_idx].copy() # set i,j entry to be the entry from precision matrix for given cluster
                #A[idx][idx2] = greatest_change_mat[idx][idx2].copy() # set i,j entry to be the entry from precision matrix for given cluster
        if len(np.nonzero(A)[0]) > 0:
            new_basis_matrices.append(vectorize_matrix(A))
    new_basis_matrices = np.array(new_basis_matrices)
    for i in range(new_basis_matrices.shape[0]):
        curr_mat = symmetrize_from_vector(new_basis_matrices[i], dim)
        nonzero_cols = np.nonzero(np.any(curr_mat != 0, axis=0))[0]
        #print("Matrix {} Len {} Channels Contained {}".format(i, len(nonzero_cols), nonzero_cols))
        #print()
    #print("NEW BASIS MATRICES SHAPE", new_basis_matrices.shape)

    return new_basis_matrices

def recurse_on_candidate(data_one, data_two, cutree1, greatest_change_mat_idx, basis_mats, dim, condition_number_thresh, recursion_min=2):
    """
    DFS recursion function

    Modified just to handle window-to-window comparisons without time axis
    """
    greatest_change_mat = symmetrize_from_vector(basis_mats[greatest_change_mat_idx], dim=dim)
    nonzero_cols = np.nonzero(np.any(greatest_change_mat != 0, axis=0))[0]
    data_total_window = np.concatenate((data_one, data_two), axis=1)
    C_one = np.cov(data_one, bias=True)
    #C_one = C_one #+ np.eye(C_one.shape[0])*1e-8
    C_two = np.cov(data_two, bias=True)
    #C_two = C_two #+ np.eye(C_two.shape[0])*1e-8 # correct numerical instability
    C_total = np.cov(data_total_window, bias=True)
    #C_total = C_total #+ np.eye(C_total.shape[0])*1e-8


    alphas_pre = optim_boyd(C=C_one, H_s=basis_mats)
    cluster_precision = greatest_change_mat[~np.all(greatest_change_mat == 0, axis=1)]
    cluster_precision = alphas_pre[greatest_change_mat_idx]*cluster_precision[:, ~np.all(cluster_precision == 0, axis=0)]

    cluster_eigvals = sorted(np.linalg.eig(cluster_precision)[0], reverse=True)
    cluster_eigval_max = max(cluster_eigvals)
    cluster_eigval_min = min(cluster_eigvals)
    eigval_ratio = cluster_eigval_max/cluster_eigval_min # condition number
    if len(nonzero_cols) >= recursion_min and eigval_ratio >= condition_number_thresh:
        new_basis_matrices = recursive_split_basis_matrix(basis_mats, greatest_change_mat_idx, dim=dim)
        nonzero_cols_one = np.nonzero(np.any(symmetrize_from_vector(new_basis_matrices[0], dim) != 0, axis=0))[0]
        if new_basis_matrices.shape[0] > 1:
            nonzero_cols_two = np.nonzero(np.any(symmetrize_from_vector(new_basis_matrices[1], dim) != 0, axis=0))[0]
            if len(nonzero_cols_one) >= recursion_min or len(nonzero_cols_two) >= recursion_min:
                cutree1[nonzero_cols_two] = int(cutree1.max() + 1)
                reduced_basis_mats = np.delete(basis_mats, greatest_change_mat_idx, axis=0)
                updated_basis_matrices = np.concatenate((reduced_basis_mats, new_basis_matrices), axis=0)
                basis_mats = updated_basis_matrices
                p_val_min, greatest_change_mat_idx = fit_optim_candidate_point(data_one=data_one, data_two=data_two, data_total=data_total_window, C_one=C_one, C_two=C_two, C_full=C_total, H_s=basis_mats, lam=0.17, dim=dim)
                return recurse_on_candidate(data_one=data_one, data_two=data_two, cutree1=cutree1, greatest_change_mat_idx=greatest_change_mat_idx, basis_mats=basis_mats, dim=dim, condition_number_thresh=condition_number_thresh, recursion_min=recursion_min)
    
    #print(basis_mats.shape)
    p_val_min, greatest_change_mat_idx = fit_optim_candidate_point(data_one=data_one, data_two=data_two, data_total=data_total_window, C_one=C_one, C_two=C_two, C_full=C_total, H_s=basis_mats, lam=0.17, dim=dim)
    
    return p_val_min, greatest_change_mat_idx


def fit_optim_candidate_point(data_one, data_two, data_total, C_one, C_two, C_full, H_s, lam, dim):
    coeffs_hat_total = optim_boyd(C=C_full, H_s=H_s)
    # null_likelihood = full_likelihood(coeffs_hat_total, H_s, C_full, N=window_size*2, 
    #                                   lam=lam, include_l1=False, debug_title='global')
    null_likelihood = full_likelihood(coeffs_hat_total, H_s, C_full, N=data_total.shape[1], 
                                        lam=lam, include_l1=False, debug_title='global')
    test_stats_m = []
    p_vals_m = []
    for k in range(H_s.shape[0]):
        # alpha_i_change_pre_temp = optim_boyd(C=C_one, H_s=H_s)
        # alpha_i_change_post_temp = optim_boyd(C=C_two, H_s=H_s)
        # alpha_i_change_pre = coeffs_hat_total.copy()
        # alpha_i_change_post = coeffs_hat_total.copy()
        # alpha_i_change_pre[k] = alpha_i_change_pre_temp[k]
        # alpha_i_change_post[k] = alpha_i_change_post_temp[k]
        alpha_i_change_pre = coeffs_hat_total.copy()
        alpha_i_change_post = coeffs_hat_total.copy()
        curr_alpha_i_pre = optim_boyd_dc(C=C_one, H=H_s[k])
        curr_alpha_i_post = optim_boyd_dc(C=C_two, H=H_s[k])
        alpha_i_change_pre[k] = curr_alpha_i_pre
        alpha_i_change_post[k] = curr_alpha_i_post
        alt_likelihood_alpha_i_pre = full_likelihood(alpha_i_change_pre, H_s, C_one, N=data_one.shape[1], 
                                                        lam=lam, include_l1=False, debug_title='Pre')
        # likelihood on post data, alpha_one change
        alt_likelihood_alpha_i_post = full_likelihood(alpha_i_change_post, H_s, C_two, N=data_two.shape[1], 
                                                        lam=lam, include_l1=False, debug_title='Post')
        alt_likelihood_alpha_i = alt_likelihood_alpha_i_pre + alt_likelihood_alpha_i_post
        #alt_likelihood_alpha_i_alt = alt_likelihood_alpha_i_pre_alt + alt_likelihood_alpha_i_post_alt
        #print("Likelihood Diff Total {}".format(alt_likelihood_alpha_i - alt_likelihood_alpha_i_alt))
        #dof = 0.5*C_full.shape[0]*(C_full.shape[0]+1) - (M + 1)
        dof = 2
        test_stat_i, p_val_i = likelihood_ratio_test(null_likelihood, 
                                            alt_likelihood_alpha_i, dof, log_pvals=1)
        test_stats_m.append(test_stat_i)
        p_vals_m.append(p_val_i)
    
    test_stat_max = np.array(test_stats_m).max()
    p_vals_corrected = meinshausen_correction(H_s=H_s, p_vals_all=np.expand_dims(np.array(p_vals_m), 0), dim=dim, log_pvals=1)

    return p_vals_corrected.min(), p_vals_corrected.squeeze().argmin()


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

def top_down_search(Z, precision, C, N, dim, clust_dist_mat):
    """
    Z: linkage matrix
    precision: precision matrix
    C: covariance matrix
    N: number of samples
    dim: dimensionality
    """

    ordered_dists = sorted(Z[:, 2], reverse=True)[1:] # skip the first distance of all one cluster
    best_basis_mats = None
    best_silhoutte_score = -np.inf
    best_cutree = None
    for dist in ordered_dists:
        #print(dist)
        cutree1 = hierarchy.fcluster(Z, t=dist, criterion='distance').squeeze()
        #print(cutree1)
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
        to_deletes = []
        singleton_idxs = []
        singleton_cluster = np.zeros(precision.shape)
        for i in range(basis_matrices.shape[0]): # iterate over clusters
            curr_mat = symmetrize_from_vector(basis_matrices[i], dim=dim)
            nonzero_cols = np.nonzero(np.any(curr_mat != 0, axis=0))[0]
            if len(nonzero_cols) <= 1:
                # copy over singletons to their own cluster
                singleton_cluster[nonzero_cols[0]][nonzero_cols[0]] = curr_mat[nonzero_cols[0]][nonzero_cols[0]]
                # delete the corresponding matrix from the basis matrices
                to_deletes.append(i)
                singleton_idxs.append(nonzero_cols[0])
        if len(to_deletes) > 0:
            basis_matrices = np.delete(basis_matrices, to_deletes, axis=0)
            basis_matrices = np.append(basis_matrices, np.expand_dims(vectorize_matrix(singleton_cluster), 0), axis=0)
            curr_idxs_min = cutree1[singleton_idxs].min()
            cutree1[singleton_idxs] = curr_idxs_min
            #cutree1 = np.delete(cutree1, singleton_idxs)
        cutree_set = list(set(cutree1))
        cutree_new = cutree1.copy()
        for i, cval in enumerate(cutree_set):
            cutree_new[cutree1 == cval] = i+1
        assert is_pos_def(symmetrize_from_vector(basis_matrices.sum(axis=0), dim=dim)), "Not PosDef"
        if len(list(set(cutree_new))) > 1:
            #clust_dist_mat_reduced = np.delete(clust_dist_mat, singleton_idxs, axis=0)
            #clust_dist_mat_reduced = np.delete(clust_dist_mat_reduced, singleton_idxs, axis=1)
            #print(clust_dist_mat_reduced.shape)
            #print(cutree_new.shape)
            #print(cutree_new)
            curr_silhoutte_score = silhouette_score(clust_dist_mat, cutree_new, metric='precomputed')
            if curr_silhoutte_score > best_silhoutte_score:
                best_basis_mats = basis_matrices.copy()
                best_silhoutte_score = curr_silhoutte_score
                best_cutree = cutree_new

    return best_basis_mats, best_cutree

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
    #pairwise_distances = sch.distance.pdist(clust_dist_mat)
    pairwise_distances = squareform(clust_dist_mat)
    Z = linkage(pairwise_distances, method='ward')
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
        if len(nonzero_cols) < 1: # don't split into a singleton cluster - not informative - 3x3 is the minimum for splitting
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
        if len(nonzero_cols) <= 2:
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
        #print("Splitting")
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
                #correction_factor = dim/C_val
                #print(correction_factor)
                # correct p_vals for cluster at all time points independently
                #p_val_i = p_val_i*correction_factor
                correction_factor = np.log(dim) - np.log(C_val)
                p_val_i = p_val_i + correction_factor

            else:
                p_val_i = 0.0
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
                #correction_factor = dim/C_val
                #print(correction_factor)
                # correct p_vals for cluster at all time points independently
                #p_val_i_true = p_val_i_true*correction_factor
                correction_factor = np.log(dim) - np.log(C_val)
                p_val_i_true = p_val_i_true + correction_factor
            else:
                p_val_i_true = 0.0
            p_vals_true_curr.append(p_val_i_true)
            lrt_vals_true_curr.append(test_stat_i_true)
            """"""

            """"""
            # est H LRT
            glasso = GraphicalLasso(max_iter=1000, alpha=5e-2, tol=1e-5, verbose=False).fit(data_one.T)
            #glasso = QuicGraphLasso(lam=5e-2).fit(data_one.T)
            precision = glasso.precision_.copy()
            clust_dist_mat = np.abs(precision)
            np.fill_diagonal(clust_dist_mat, 0.0)
            ###########
            clust_dist_mat = (clust_dist_mat.max()+1e-5) - clust_dist_mat
            np.fill_diagonal(clust_dist_mat, 0.0)
            ###########
            #pairwise_distances = sch.distance.pdist(clust_dist_mat)
            pairwise_distances = squareform(clust_dist_mat)
            Z = linkage(pairwise_distances, method='ward')
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

def collect_prec_rec_results(M, lam, dims=[20,40,60,80], linkage_type='single', plotting=False):
    # collect results
    post_window_size = 20
    w = post_window_size
    seed_list = np.arange(0, 50)
    coeff_change_mags = np.round(np.arange(0.0, 1.0, 0.1), 2).tolist()
    dim_results = {}
    #dims = [20, 40, 60, 80]
    #dims = [60]
    dim_counter = 0
    for dim in dims:
        #M = M + 2*dim_counter # scale M with the dimensionality
        H_s, prec_temp, _ = generate_matrices_orthogonal(M=M, dim=dim, to_print=False, lam=lam, linkage_type=linkage_type)
        H_s_true = H_s.copy()
        first_prec_coeffs = np.ones(M)
        prec_one = collect_precision_matrix(H_s=H_s_true, prec_coeffs=first_prec_coeffs, P=dim)
        print("Max Precision Entry", np.abs(np.triu(prec_one, k=1)).max())
        print("Min Precision Entry", np.abs(np.triu(prec_one, k=1)).min())
        print("Sparsity {}".format((prec_one == 0).sum()/len(prec_one.flatten())))
        total_results = {}
        dim_precisions = []
        dim_f1 = []
        dim_recall = []
        dim_accuracy = []
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
            max_coeff_diffs_curr = []
            max_likelihood_improvements_curr = []
            cai_test_vals = []
            p_vals_curr_dfs = []
            mag_results = {}
            for curr_seed in tqdm(seed_list):
                np.random.seed(curr_seed)
                second_prec_coeffs = first_prec_coeffs.copy()
                changed_coeff_idx = np.random.choice(np.arange(M))
                second_prec_coeffs[changed_coeff_idx] = second_prec_coeffs[changed_coeff_idx] + curr_change_mag # make it additive so we don't have PSD problems
                prec_two = collect_precision_matrix(H_s=H_s, prec_coeffs=second_prec_coeffs, P=dim)
                #data_one, _ = sim_data(covar=inv(prec_one), dim=dim, N=200)
                if dim == 20:
                    data_one_train, _ = sim_data(covar=inv(prec_one), dim=dim, N=50) # identical to data_one, used for fitting the initial Glasso/Clime estimates
                elif dim == 40:
                    data_one_train, _ = sim_data(covar=inv(prec_one), dim=dim, N=100) # identical to data_one, used for fitting the initial Glasso/Clime estimates
                elif dim == 60:
                    data_one_train, _ = sim_data(covar=inv(prec_one), dim=dim, N=150) # identical to data_one, used for fitting the initial Glasso/Clime estimates
                elif dim == 80:
                    data_one_train, _ = sim_data(covar=inv(prec_one), dim=dim, N=200) # identical to data_one, used for fitting the initial Glasso/Clime estimates
                if curr_change_mag > 0.0:
                    data_two, _ = sim_data(covar=inv(prec_two), dim=dim, N=post_window_size)
                else:
                    data_two, _ = sim_data(covar=inv(prec_one), dim=dim, N=post_window_size)
                data_full = np.concatenate((data_one_train, data_two), axis=1)
                data_full_train = np.concatenate((data_one_train, data_two), axis=1)
                scaler = StandardScaler().fit(data_one_train.T)
                data_one_train = scaler.transform(data_one_train.T).T
                data_two = scaler.transform(data_two.T).T
                data_full_train = scaler.transform(data_full_train.T).T
                C_one = np.cov(data_one_train, bias=True)
                C_one_train = np.cov(data_one_train, bias=True)
                C_two = np.cov(data_two, bias=True)
                C_full = np.cov(data_full_train, bias=True)
                C_full_train = np.cov(data_full_train, bias=True)
                #coeffs_hat_total = optim_boyd(C=C_full, H_s=H_s_true)
                coeffs_hat_total = optim_boyd(C=C_full_train, H_s=H_s_true)
                #alpha_one = optim_boyd(C=C_one, H_s=H_s_true)
                #alpha_two = optim_boyd(C=C_two, H_s=H_s_true)

                r_max = np.max(np.abs(C_one_train[np.triu_indices(dim, k=1)]))
                #lambda_search = [0.1, 0.13, 0.17, 0.185, 0.2, 0.22, 0.26]
                lambda_search = [0.05 + (j*(r_max-0.05))/40 for j in range(1, 41, 1)]
                """"""
                # given true H's, estimate coefficients
                alpha_i_change_pre = coeffs_hat_total.copy()
                alpha_i_change_post = coeffs_hat_total.copy()
                curr_alpha_i_pre = optim_boyd_dc(C=C_one, H=H_s_true[changed_coeff_idx])
                curr_alpha_i_post = optim_boyd_dc(C=C_two, H=H_s_true[changed_coeff_idx])
                alpha_i_change_pre[changed_coeff_idx] = curr_alpha_i_pre
                alpha_i_change_post[changed_coeff_idx] = curr_alpha_i_post
                null_likelihood = full_likelihood(coeffs_hat_total, H_s_true, C_full_train, N=data_full_train.shape[1], 
                                                    lam=lam, include_l1=False, debug_title='global')
                # # likelihood on pre data, alpha_one change
                alt_likelihood_alpha_i_pre = full_likelihood(alpha_i_change_pre, H_s_true, C_one_train, N=data_one_train.shape[1], 
                                                                lam=lam, include_l1=False, debug_title='Pre')
                # likelihood on post data, alpha_one change
                alt_likelihood_alpha_i_post = full_likelihood(alpha_i_change_post, H_s_true, C_two, N=data_two.shape[1], 
                                                                lam=lam, include_l1=False, debug_title='Post')
                alt_likelihood_alpha_i = alt_likelihood_alpha_i_pre + alt_likelihood_alpha_i_post
                dof = 2
                test_stat_i, p_val_i = likelihood_ratio_test(null_likelihood, 
                                                    alt_likelihood_alpha_i, dof, log_pvals=1)
                curr_mat = symmetrize_from_vector(H_s_true[changed_coeff_idx], dim)
                nonzero_cols = np.nonzero(np.any(curr_mat != 0, axis=0))[0]
                C_val = len(nonzero_cols)
                if C_val > 0:
                    #correction_factor = dim/C_val
                    #print(correction_factor)
                    # correct p_vals for cluster at all time points independently
                    #p_val_i = p_val_i*correction_factor
                    correction_factor = np.log(dim) - np.log(C_val)
                    p_val_i = p_val_i + correction_factor
                else:
                    p_val_i = 0.0
                p_vals_curr.append(p_val_i)
                lrt_vals_curr.append(test_stat_i)
                coeff_diffs_curr.append(alpha_i_change_pre[changed_coeff_idx]-alpha_i_change_post[changed_coeff_idx])
                """"""

                """"""
                # true LRT
                null_likelihood_true = full_likelihood(coeffs_hat_total, H_s_true, C_full_train, N=data_full_train.shape[1], 
                                                    lam=lam, include_l1=False, debug_title='global')
                # # likelihood on pre data, alpha_one change
                alt_likelihood_alpha_i_pre_true = full_likelihood(first_prec_coeffs, H_s_true, C_one_train, N=data_one_train.shape[1], 
                                                                lam=lam, include_l1=False, debug_title='Pre')
                # likelihood on post data, alpha_one change
                alt_likelihood_alpha_i_post_true = full_likelihood(second_prec_coeffs, H_s_true, C_two, N=data_two.shape[1], 
                                                                lam=lam, include_l1=False, debug_title='Post')
                alt_likelihood_alpha_i_true = alt_likelihood_alpha_i_pre_true + alt_likelihood_alpha_i_post_true
                dof = 2
                test_stat_i_true, p_val_i_true = likelihood_ratio_test(null_likelihood_true, 
                                                    alt_likelihood_alpha_i_true, dof, log_pvals=1)
                curr_mat = symmetrize_from_vector(H_s_true[changed_coeff_idx], dim)
                nonzero_cols = np.nonzero(np.any(curr_mat != 0, axis=0))[0]
                C_val = len(nonzero_cols)
                if C_val > 0:
                    #correction_factor = dim/C_val
                    #print(correction_factor)
                    # correct p_vals for cluster at all time points independently
                    #p_val_i_true = p_val_i_true*correction_factor
                    correction_factor = np.log(dim) - np.log(C_val)
                    p_val_i_true = p_val_i_true + correction_factor
                else:
                    p_val_i_true = 0.0
                p_vals_true_curr.append(p_val_i_true)
                lrt_vals_true_curr.append(test_stat_i_true)
                """"""

                """"""
                # est H LRT

                precision, chosen_lamb = thav_gl_fn(data_train=data_one_train, lambda_search=lambda_search, C=0.5, threshold=1.0)
                #lamb_search_new = [chosen_lamb + j/100 for j in range(1, 8, 1)]
                #lamb_search_new.append(chosen_lamb)
                #lamb_search_new = sorted(lamb_search_new)
                #lamb_search_new = [chosen_lamb+x for x in np.arange(chosen_lamb, chosen_lamb+0.2, 0.025)]
                #precision, chosen_lamb = thav_gl_fn(data_train=data_one_train, lambda_search=lamb_search_new, C=0.5, threshold=1.0)
                #print(chosen_lamb)
                """"""
                #glasso = GraphicalLasso(max_iter=1000, alpha=5e-2, tol=1e-5, verbose=False).fit(data_one.T)
                #glasso = QuicGraphicalLasso(max_iter=500, lam=5e-2, tol=1e-4).fit(data_one_train.T)
                #glasso = GraphicalLasso(max_iter=1500, alpha=lam, tol=1e-4, verbose=False).fit(data_one_train.T) # this one has more data used

                # glasso = GraphicalLassoCV(alphas=lambda_search, n_refinements=4, tol=1e-4, max_iter=1500, cv=5).fit(data_one_train.T)
                # chosen_lamb = glasso.alpha_
                # #lamb_search_new = [chosen_lamb+x for x in np.arange(chosen_lamb-0.02, chosen_lamb+0.2, 0.01)]
                # lamb_search_new = [chosen_lamb + j/100 for j in range(1, 6, 1)]
                # lamb_search_new.append(chosen_lamb)
                # lamb_search_new = sorted(lamb_search_new)
                # glasso = GraphicalLassoCV(alphas=lamb_search_new, n_refinements=4, tol=1e-4, max_iter=1500, cv=5).fit(data_one_train.T)
                # precision = glasso.precision_.copy()
                #precision[np.abs(precision) <= 0.8*1.8*chosen_lamb] = 0.0
                precision[np.abs(precision) <= 0.1] = 0.0

                """
                Plotting
                """
                if plotting:
                    sns.heatmap(prec_one)
                    plt.savefig('./debugging_figs/heatmap_true.png')
                    plt.close()
                    prec_normalized = precision.copy()
                    for i in range(precision.shape[1]):
                        prec_normalized[:, i] = precision[:, i]/precision[i][i]
                    sns.heatmap(precision)
                    plt.savefig('./debugging_figs/heatmap_est.png')
                    plt.close()
                """
                End Plotting
                """
                

                
                #print("Sparsity Est {}".format((precision == 0).sum()/len(precision.flatten())))
                """"""
                first_idces = np.sign(np.abs(precision[np.triu_indices(dim, k=1)]))
                second_indces = np.sign(np.abs(prec_one[np.triu_indices(dim, k=1)]))
                curr_f1 = f1_score(y_true=second_indces, y_pred=first_idces)
                curr_precision = precision_score(y_true=second_indces, y_pred=first_idces)
                curr_recall = recall_score(y_true=second_indces, y_pred=first_idces)
                curr_acc = accuracy_score(y_true=second_indces, y_pred=first_idces)
                #print(curr_f1, curr_precision, curr_recall, curr_acc)
                dim_f1.append(curr_f1)
                dim_precisions.append(curr_precision)
                dim_recall.append(curr_recall)
                dim_accuracy.append(curr_acc)
                #print(first_idces)
                ##print(second_indces)
                #print((first_idces == second_indces).sum()/len(precision[np.triu_indices(dim, k=1)].flatten()))
                clust_dist_mat = np.abs(precision)
                np.fill_diagonal(clust_dist_mat, 0.0)
                ###########
                #clust_dist_mat = 1.0 - clust_dist_mat
                clust_dist_mat = (clust_dist_mat.max()+1e-5) - clust_dist_mat
                np.fill_diagonal(clust_dist_mat, 0.0)
                #clust_dist_mat[clust_dist_mat == clust_dist_mat.max()] = 1e8
                ###########
                #pairwise_distances = sch.distance.pdist(clust_dist_mat)
                pairwise_distances = squareform(clust_dist_mat)
                Z = linkage(pairwise_distances, method=linkage_type)
                """
                Plotting
                """
                if plotting:
                    plt.figure()
                    dn = hierarchy.dendrogram(Z)
                    plt.savefig(os.path.join('debugging_figs', "dendrogram.png"))
                    plt.close()
                """
                End Plotting
                """
                #cutree1 = hierarchy.cut_tree(Z, n_clusters=M).squeeze()


                """
                Top Down Search goes here
                """
                basis_matrices, cutree1 = top_down_search(Z=Z, precision=precision, C=C_one_train, N=data_one_train.shape[0], dim=dim, clust_dist_mat=clust_dist_mat)
                """
                """

                # cutree1 = hierarchy.fcluster(Z, t=clust_distances[-M], criterion='distance')
                # basis_matrices = []
                # for i in range(max(set(cutree1))+1): # iterate over clusters
                #     idxs = np.where(cutree1 == i)[0] # indexes for given cluster
                #     A = np.zeros(precision.shape) # blank A matrix
                #     for idx in idxs: # loop over indexes
                #         for idx2 in idxs: # loop over indexes
                #             A[idx][idx2] = precision[idx][idx2].copy() # set i,j entry to be the entry from precision matrix for given cluster
                #     if len(np.nonzero(A)[0]) > 0:
                #         basis_matrices.append(vectorize_matrix(A))
                # basis_matrices = np.array(basis_matrices)
                # curr_silhoutte_score = silhouette_score(clust_dist_mat, cutree1, metric='precomputed')

                #basis_matrices = bfs_basis_mats(data_one_train, basis_matrices, curr_silhoutte_score, dim=dim, recursion_min=1, cutree=cutree1, root_dist_mat=clust_dist_mat)
                assert is_pos_def(symmetrize_from_vector(basis_matrices.sum(axis=0), dim=dim)), "Not PosDef"

                """
                Plotting
                """
                if plotting:
                    for idx in range(basis_matrices.shape[0]):
                        sns.heatmap(symmetrize_from_vector(basis_matrices[idx], dim=dim))
                        plt.savefig('./debugging_figs/heatmap_h{}.png'.format(idx))
                        plt.close()
                    exit()
                """
                End Plotting
                """
                


                # for i in range(H_s_true.shape[0]):
                #     curr_mat = symmetrize_from_vector(H_s_true[i], dim)
                #     #curr_mat = H_s[i]
                #     nonzero_cols = np.nonzero(np.any(curr_mat != 0, axis=0))[0]
                #     print("****************************************")
                #     print("SIMULATION MATRICES")
                #     if i == (H_s_true.shape[0] - 1):
                #         print("Sim Basis Matrix {}".format(i))
                #     else:
                #         print("Sim Basis Matrix {}".format(i))
                #     print("Sim Channels Contained {}".format(nonzero_cols))
                #     #print("Diag: ", set(list(np.diag(curr_mat))))
                #     #print("OffDiag: ", set(list(curr_mat[np.triu_indices(dim, k=1)])))
                #     print("****************************************")
                #     print()
                # for i in range(basis_matrices.shape[0]):
                #     curr_mat = symmetrize_from_vector(basis_matrices[i], dim)
                #     #curr_mat = H_s[i]
                #     nonzero_cols = np.nonzero(np.any(curr_mat != 0, axis=0))[0]
                #     print("****************************************")
                #     print("EST MATRICES")
                #     if i == (basis_matrices.shape[0] - 1):
                #         print("Est Basis Matrix {}".format(i))
                #     else:
                #         print("Est Basis Matrix {}".format(i))
                #     print("Est Channels Contained {}".format(nonzero_cols))
                #     #print("Diag: ", set(list(np.diag(curr_mat))))
                #     #print("OffDiag: ", set(list(curr_mat[np.triu_indices(dim, k=1)])))
                #     print("****************************************")
                #     print()
                # print("True Precision Matrix", prec_one[0, :]) 
                # print("Glasso Precision Matrix", precision[0, :])
                # # print(np.abs(prec_one[0, :] - precision[0, :]))
                # # print()
                # # print()
                # print(symmetrize_from_vector(H_s_true[0], dim=dim)[0, :])
                # print(symmetrize_from_vector(basis_matrices[0], dim=dim)[0, :])
                # # print(symmetrize_from_vector(np.abs(H_s_true[0] - basis_matrices[0]), dim=dim)[0, :])
                # exit()

                #coeffs_hat_total = optim_boyd(C=C_full_train, H_s=basis_matrices)
                coeffs_hat_total_train = optim_boyd(C=C_full_train, H_s=basis_matrices)
                p_val_min = np.inf
                test_stat_max = 0
                max_coeff_diff = 0
                max_likelihood_improvement = 0
                chosen_C_val = 0
                p_val_argmin = 0
                #print("First", first_prec_coeffs)
                #print("Second", second_prec_coeffs)
                #print("Total", coeffs_hat_total_train)
                for i in range(basis_matrices.shape[0]):
                    alpha_i_change_pre = coeffs_hat_total_train.copy()
                    alpha_i_change_post = coeffs_hat_total_train.copy()
                    curr_alpha_i_pre = optim_boyd_dc(C=C_one_train, H=basis_matrices[i], iters=100)
                    curr_alpha_i_post = optim_boyd_dc(C=C_two, H=basis_matrices[i], iters=100)
                    alpha_i_change_pre[i] = curr_alpha_i_pre
                    alpha_i_change_post[i] = curr_alpha_i_post
                    #print("Pre", alpha_i_change_pre)
                    #print("Post", alpha_i_change_post)
                    null_likelihood = full_likelihood(coeffs_hat_total_train, basis_matrices, C_full_train, N=data_full_train.shape[1], 
                                                        lam=lam, include_l1=False, debug_title='global')
                    # # likelihood on pre data, alpha_one change
                    alt_likelihood_alpha_i_pre = full_likelihood(alpha_i_change_pre, basis_matrices, C_one_train, N=data_one_train.shape[1], 
                                                                    lam=lam, include_l1=False, debug_title='Pre')
                    # likelihood on post data, alpha_one change
                    alt_likelihood_alpha_i_post = full_likelihood(alpha_i_change_post, basis_matrices, C_two, N=data_two.shape[1], 
                                                                    lam=lam, include_l1=False, debug_title='Post')
                    alt_likelihood_alpha_i = alt_likelihood_alpha_i_pre + alt_likelihood_alpha_i_post
                    
                    dof = 2
                    test_stat_i, p_val_i = likelihood_ratio_test(null_likelihood, 
                                                        alt_likelihood_alpha_i, dof, log_pvals=1)
                    curr_mat = symmetrize_from_vector(basis_matrices[i], dim)
                    nonzero_cols = np.nonzero(np.any(curr_mat != 0, axis=0))[0]
                    C_val = len(nonzero_cols)
                    if C_val > 0:
                        #correction_factor = dim/C_val
                        #print(correction_factor)
                        # correct p_vals for cluster at all time points independently
                        #p_val_i = p_val_i*correction_factor
                        correction_factor = np.log(dim) - np.log(C_val)
                        p_val_i = p_val_i + correction_factor
                    else:
                        p_val_i = 0.0
                    if p_val_i < p_val_min:
                        p_val_min = p_val_i
                        test_stat_max = test_stat_i
                        chosen_C_val = C_val
                        p_val_argmin = i

                    coeff_diff = np.abs(alpha_i_change_pre - alpha_i_change_post).max()
                    if coeff_diff > max_coeff_diff:
                        max_coeff_diff = coeff_diff
                    likelihood_improvement = alt_likelihood_alpha_i - null_likelihood
                    if likelihood_improvement > max_likelihood_improvement:
                        max_likelihood_improvement = likelihood_improvement
                    #print("\n", p_val_i, test_stat_i,"\n")
                #print("Pval {} Cluster Size {}".format(p_val_min, chosen_C_val))
                p_vals_curr_est.append(p_val_min)
                lrt_vals_curr_est.append(test_stat_max)
                coeff_diffs_curr_est.append(alpha_i_change_pre[changed_coeff_idx]-alpha_i_change_post[changed_coeff_idx]) # this shouldn't be used for anything really
                max_coeff_diffs_curr.append(max_coeff_diff)
                max_likelihood_improvements_curr.append(max_likelihood_improvement)
                """"""


                """
                DFS Recursion
                """
                p_val_min_dfs, _ = recurse_on_candidate(data_one=data_one_train, data_two=data_two, cutree1=cutree1, greatest_change_mat_idx=p_val_argmin, basis_mats=basis_matrices, dim=dim, condition_number_thresh=5.0, recursion_min=2)
                p_vals_curr_dfs.append(p_val_min_dfs)
                """"""

                #clime_init = clime_init_fn(5e-2, data_one.T)
                clime_init = clime_init_fn(chosen_lamb, data_one_train.T) # this one has more data used
                g1 = calc_g1(w)
                g2 = calc_g2(w)
                rhat0 = calc_rhat(clime_init, dim)
                T0 = calc_T_t(X=data_two.T, omega_hat=clime_init, r_hat=rhat0, w=w, p=dim, t=0, g1=g1, g2=g2)
                test_vals_kesh_curr.append(T0)

                clime_init_true = prec_one.copy()
                rhat0_true = calc_rhat(clime_init_true, dim)
                T0_true = calc_T_t(X=data_two.T, omega_hat=clime_init_true, r_hat=rhat0_true, w=w, p=dim, t=0, g1=g1, g2=g2)
                test_vals_kesh_true_curr.append(T0_true)



                """
                Cai Algorithm
                """
                residuals_x, residuals_y, beta_hats_x, beta_hats_y  = perform_regression(X_data=data_one_train.T, Y_data=data_two.T)
                residuals_x_cov_corrected = bias_corrected_residual_covariance(residuals_x, beta_hats_x)
                T_x = calculate_T(residuals_x_cov_corrected)
                theta_x = calculate_theta(residuals_x_cov_corrected, beta_hats_x, N=data_one_train.shape[1])

                residuals_y_cov_corrected = bias_corrected_residual_covariance(residuals_y, beta_hats_y)
                T_y = calculate_T(residuals_y_cov_corrected)
                theta_y = calculate_theta(residuals_y_cov_corrected, beta_hats_y, N=data_two.shape[1])

                W_cai = calculate_standardized_stat(T_x, T_y, theta_x, theta_y)
                M_cai = calculate_global_stat(W_cai)
                
                cai_test_vals.append(M_cai)
                """"""
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
            mag_results['likelihood_improvement'] = np.array(max_likelihood_improvements_curr)
            mag_results['max_coeff_diffs'] = np.array(max_coeff_diffs_curr)
            mag_results['cai_test'] = np.array(cai_test_vals)
            mag_results['our_p_vals_dfs'] = np.array(p_vals_curr_dfs)
            total_results[curr_change_mag] = mag_results
            #print(mag_results['our_p_vals'].shape, mag_results['our_p_vals_true'].shape, mag_results['our_p_vals_est'].shape)
        dim_results[dim] = total_results
        dim_counter += 1
        print("Dim {} F1 {}".format(dim, np.array(dim_f1).mean()))
        print("Dim {} Prec {}".format(dim, np.array(dim_precisions).mean()))
        print("Dim {} Rec {}".format(dim, np.array(dim_recall).mean()))
        print("Dim {} Acc {}".format(dim, np.array(dim_accuracy).mean()))
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

def plot_precision_recall_comparison(prec_ours, rec_ours, prec_other, rec_other, dim, labl='Ours', third_prec=None, third_rec=None, dfs_prec=None, dfs_rec=None):
    plt.plot(rec_ours, prec_ours, label='Ours')
    plt.plot(rec_other, prec_other, label='Kesh')
    if third_prec is not None and third_rec is not None:
        plt.plot(third_rec, third_prec, label='Cai')
    if dfs_prec is not None and dfs_rec is not None:
        plt.plot(dfs_rec, dfs_prec, label='Ours + DFS')
    
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

def process_prec_rec_results(dim_results, dims=[20, 40, 60, 80]):
    print("Plotting Results...")
    seed_list = np.arange(0, 50)
    coeff_change_mags = np.round(np.arange(0.1, 1.0, 0.1), 2).tolist() # just iterate over the actual changes, compare vs no change
    all_coeff_change_mags = np.round(np.arange(0.0, 1.0, 0.1), 2).tolist()
    #dims = [20, 40, 60, 80]
    for dim in dims:
        curr_dim_no_change_ours = dim_results[dim][0.0]['our_p_vals'] # with H given
        curr_dim_no_change_ours_est = dim_results[dim][0.0]['our_p_vals_est'] # nothing given
        curr_dim_no_change_ours_true = dim_results[dim][0.0]['our_p_vals_true'] # everything given
        curr_dim_no_change_ours_dfs = dim_results[dim][0.0]['our_p_vals_dfs'] # nothing given

        curr_dim_no_change_lrt_est = dim_results[dim][0.0]['our_lrt_est'] # nothing given

        curr_dim_no_change_kesh = dim_results[dim][0.0]['kesh_test'] # with nothing given
        curr_dim_no_change_kesh_true = dim_results[dim][0.0]['kesh_test_true'] # everything given
        curr_dim_no_change_coeff_diff = dim_results[dim][0.0]['max_coeff_diffs']
        curr_dim_no_change_likelihood_improvement = dim_results[dim][0.0]['likelihood_improvement']

        curr_dim_no_change_cai = dim_results[dim][0.0]['cai_test'] # with nothing given
        nc_lrt_avg = curr_dim_no_change_lrt_est.mean()
        nc_pval_avg = curr_dim_no_change_ours_est.mean()
        nc_pval_max = curr_dim_no_change_ours_est.max()
        nc_pval_min = curr_dim_no_change_ours_est.min()
        nc_pval_std = curr_dim_no_change_ours_est.std()
        nc_pval_median = np.median(curr_dim_no_change_ours_est)

        nc_coeff_diff_avg = curr_dim_no_change_coeff_diff.mean()
        nc_coeff_diff_max = curr_dim_no_change_coeff_diff.max()
        nc_coeff_diff_min = curr_dim_no_change_coeff_diff.min()
        nc_coeff_diff_std = curr_dim_no_change_coeff_diff.std()
        nc_coeff_diff_median = np.median(curr_dim_no_change_coeff_diff)

        nc_likelihood_avg = curr_dim_no_change_likelihood_improvement.mean()
        nc_likelihood_max = curr_dim_no_change_likelihood_improvement.max()
        nc_likelihood_min = curr_dim_no_change_likelihood_improvement.min()
        nc_likelihood_std = curr_dim_no_change_likelihood_improvement.std()
        nc_likelihood_median = np.median(curr_dim_no_change_likelihood_improvement)
        print("No Change LRT Avg {} Pval Avg {} Pval Max {} Pval Min {} Pval Std {} Pval Median {}".format(nc_lrt_avg, nc_pval_avg, nc_pval_max, nc_pval_min, nc_pval_std, nc_pval_median))
        print("No Change Coeff Avg {} Max {} Min {} Std {} Median {}".format(nc_coeff_diff_avg, nc_coeff_diff_max, nc_coeff_diff_min, nc_coeff_diff_std, nc_coeff_diff_median))
        print("No Change Likelihood Avg {} Max {} Min {} Std {} Median {}".format(nc_likelihood_avg, nc_likelihood_max, nc_likelihood_min, nc_likelihood_std, nc_likelihood_median))
        for mag in tqdm(coeff_change_mags):
            curr_dim_change_ours = dim_results[dim][mag]['our_p_vals'] # with H given
            curr_dim_change_ours_est = dim_results[dim][mag]['our_p_vals_est'] # nothing given
            curr_dim_change_ours_true = dim_results[dim][mag]['our_p_vals_true'] # everything given
            curr_dim_change_ours_dfs = dim_results[dim][mag]['our_p_vals_dfs'] # nothing given

            curr_dim_change_lrt_est = dim_results[dim][mag]['our_lrt_est'] # nothing given

            curr_dim_change_likelihood_improvement = dim_results[dim][mag]['likelihood_improvement']
            curr_dim_change_coeff_diff = dim_results[dim][mag]['max_coeff_diffs']
            c_lrt_avg = curr_dim_change_lrt_est.mean()
            c_pval_avg = curr_dim_change_ours.mean()
            c_pval_max = curr_dim_change_ours.max()
            c_pval_min = curr_dim_change_ours.min()
            c_pval_std = curr_dim_change_ours.std()
            c_pval_median = np.median(curr_dim_change_ours)

            c_coeff_diff_avg = curr_dim_change_coeff_diff.mean()
            c_coeff_diff_max = curr_dim_change_coeff_diff.max()
            c_coeff_diff_min = curr_dim_change_coeff_diff.min()
            c_coeff_diff_std = curr_dim_change_coeff_diff.std()
            c_coeff_diff_median = np.median(curr_dim_change_coeff_diff)

            c_likelihood_avg = curr_dim_change_likelihood_improvement.mean()
            c_likelihood_max = curr_dim_change_likelihood_improvement.max()
            c_likelihood_min = curr_dim_change_likelihood_improvement.min()
            c_likelihood_std = curr_dim_change_likelihood_improvement.std()
            c_likelihood_median = np.median(curr_dim_change_likelihood_improvement)
            print("Change {} LRT Avg {} Pval Avg {} Pval Max {} Pval Min {} Pval Std {} Pval Median {}".format(mag, c_lrt_avg, c_pval_avg, c_pval_max, c_pval_min, c_pval_std, c_pval_median))
            print("Change Coeff Avg {} Max {} Min {} Std {} Median {}".format(c_coeff_diff_avg, c_coeff_diff_max, c_coeff_diff_min, c_coeff_diff_std, c_coeff_diff_median))
            print("Change Likelihood Avg {} Max {} Min {} Std {} Median {}".format(c_likelihood_avg, c_likelihood_max, c_likelihood_min, c_likelihood_std, c_likelihood_median))
            curr_dim_change_kesh = dim_results[dim][mag]['kesh_test'] # with nothing given
            curr_dim_change_kesh_true = dim_results[dim][mag]['kesh_test_true'] # everything given

            curr_dim_change_cai = dim_results[dim][mag]['cai_test'] # with nothing given

            prec_ours, rec_ours, fprate_ours = process_curr_pr(change_vals=curr_dim_change_ours, no_change_vals=curr_dim_no_change_ours)
            prec_ours_est, rec_ours_est, fprate_ours_est = process_curr_pr(change_vals=curr_dim_change_ours_est, no_change_vals=curr_dim_no_change_ours_est)
            prec_ours_dfs, rec_ours_dfs, fprate_ours_dfs = process_curr_pr(change_vals=curr_dim_change_ours_dfs, no_change_vals=curr_dim_no_change_ours_dfs)
            prec_ours_true, rec_ours_true, fprate_ours_true = process_curr_pr(change_vals=curr_dim_change_ours_true, no_change_vals=curr_dim_no_change_ours_true)

            prec_kesh, rec_kesh, fprate_kesh = process_curr_pr(change_vals=curr_dim_change_kesh, no_change_vals=curr_dim_no_change_kesh, pvals=False)
            prec_kesh_true, rec_kesh_true, fprate_kesh_true = process_curr_pr(change_vals=curr_dim_change_kesh_true, no_change_vals=curr_dim_no_change_kesh_true, pvals=False)

            prec_cai, rec_cai, fprate_cai = process_curr_pr(change_vals=curr_dim_change_cai, no_change_vals=curr_dim_no_change_cai, pvals=False)

            plot_precision_recall(prec_ours, rec_ours, dim=dim, labl='Ours_Magnitude_{}'.format(mag))
            plot_precision_recall(prec_ours_est, rec_ours_est, dim=dim, labl='Ours_Est_Magnitude_{}'.format(mag))
            plot_precision_recall(prec_ours_dfs, rec_ours_dfs, dim=dim, labl='Ours_DFS_Magnitude_{}'.format(mag))
            plot_precision_recall(prec_ours_true, rec_ours_true, dim=dim, labl='Ours_True_Magnitude_{}'.format(mag))

            plot_precision_recall(prec_kesh, rec_kesh, dim=dim, labl='Kesh_Magnitude_{}'.format(mag))
            plot_precision_recall(prec_kesh_true, rec_kesh_true, dim=dim, labl='Kesh_True_Magnitude_{}'.format(mag))

            plot_precision_recall(prec_cai, rec_cai, dim=dim, labl='Cai_Magnitude_{}'.format(mag))

            plot_roc(rec_ours, fprate_ours, dim=dim, labl='Ours_Magnitude_{}'.format(mag))
            plot_roc(rec_ours_est, fprate_ours_est, dim=dim, labl='Ours_Est_Magnitude_{}'.format(mag))
            plot_roc(rec_ours_dfs, fprate_ours_dfs, dim=dim, labl='Ours_DFS_Magnitude_{}'.format(mag))
            plot_roc(rec_ours_true, fprate_ours_true, dim=dim, labl='Ours_True_Magnitude_{}'.format(mag))

            plot_roc(rec_kesh, fprate_kesh, dim=dim, labl='Kesh_Magnitude_{}'.format(mag))
            plot_roc(rec_kesh_true, fprate_kesh_true, dim=dim, labl='Kesh_True_Magnitude_{}'.format(mag))

            plot_roc(rec_cai, fprate_cai, dim=dim, labl='Cai_Magnitude_{}'.format(mag))

            # comparisons, the important part
            plot_precision_recall_comparison(prec_ours, rec_ours, prec_kesh, rec_kesh, dim=dim, labl='Ours_Kesh_Magnitude_{}'.format(mag))
            plot_roc_comparison(tprate_ours=rec_ours, fprate_ours=fprate_ours, tprate_other=rec_kesh, fprate_other=fprate_kesh, dim=dim, labl='Ours_Kesh_Magnitude_{}'.format(mag))

            plot_precision_recall_comparison(prec_ours_est, rec_ours_est, prec_kesh, rec_kesh, dim=dim, labl='PR_Comparison_Magnitude_{}'.format(mag), third_prec=prec_cai, third_rec=rec_cai, dfs_prec=prec_ours_dfs, dfs_rec=rec_ours_dfs)
            plot_roc_comparison(tprate_ours=rec_ours_est, fprate_ours=fprate_ours_est, tprate_other=rec_kesh, fprate_other=fprate_kesh, dim=dim, labl='Ours_Estimate_Kesh_Magnitude_{}'.format(mag))

            plot_precision_recall_comparison(prec_ours_true, rec_ours_true, prec_kesh_true, rec_kesh_true, dim=dim, labl='Ours_True_Kesh_Magnitude_{}'.format(mag))
            plot_roc_comparison(tprate_ours=rec_ours_true, fprate_ours=fprate_ours_true, tprate_other=rec_kesh_true, fprate_other=fprate_kesh_true, dim=dim, labl='Ours_True_Kesh_Magnitude_{}'.format(mag))

        plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"])
        plt.xticks(rotation=45, ha='right')
        ylim_max = 10.0
        for i in range(1, 11):
            #print(dim_results[dim][all_coeff_change_mags[i-1]]['est_diff'])
            plot_confidence_interval(x=i, values=dim_results[dim][all_coeff_change_mags[i-1]]['est_diff'], z=1.96, color='#2187bb', horizontal_line_width=0.25)
        #plt.ylim(0, ylim_max)
        plt.title("Confidence Intervals {}".format("Differences"))
        plt.ylabel("Estimated Difference")
        plt.xlabel("Coefficient Difference")
        plt.tight_layout()
        plt.savefig('lrt_test_figs/{}'.format("conf_int_diff_total_est_{}.png".format(dim)))
        plt.close()
    

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
    args.dims = list(map(int, args.dims.split(',')))
    if args.prec_rec:
        print("Collecting Prec/Rec results")
        if args.load_results:
            with open('lrt_test_figs/dim_results.pkl', 'rb') as fp:
                res = pickle.load(fp)
                # for dim in args.dims:
                #     import copy
                #     currkeys = [float(x) for x in res[dim].keys()]
                #     print(currkeys)
                #     for k in currkeys:
                #         vals = res[dim].pop(str(k))
                #         res[dim][round(k, 2)] = vals
                print("> Results Loaded! <")            
        else:
            res = collect_prec_rec_results(M=args.M, lam=args.lam, dims=args.dims, linkage_type=args.linkage)
        prec_rec_res = process_prec_rec_results(dim_results=res, dims=args.dims)
    else:
        res = collect_test_results(M=args.M, dim=args.dim, w=args.w)
        plot_results(res)
    

    


if __name__ == '__main__':
    args = get_args()
    main(args)