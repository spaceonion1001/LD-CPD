import numpy as np
from numpy.linalg import inv as inv
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.extmath import fast_logdet
from optim import optim_boyd, unbiased_init_precision, solve_optim_global, create_global_problem, optim_boyd_dc
from likelihood import likelihood_ratio_test, full_likelihood
from scipy.stats import chi2, multivariate_normal
#from torch import zero_
sns.set()

from sklearn.covariance import graphical_lasso, GraphicalLasso, GraphicalLassoCV
from sklearn.metrics import silhouette_score, pairwise_distances as pairwise_d
from tqdm import tqdm

from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.cluster.hierarchy as sch
import scipy

from thav_gl import thav_gl_fn


from statsmodels.tsa.seasonal import STL

from utils import is_symmetric, is_pos_def, vectorize_matrix, symmetrize_from_vector
from likelihood import LRT_all_coeffs, LRT_all_coeffs_full_likelihood, LRT_individual_coeffs_full_likelihood, apply_fdr_correction, LRT_covariance
from likelihood import apply_bonferroni_correction
from meinshausen import meinshausen_correction
import warnings
warnings.filterwarnings('ignore')  # <- remember to comment this if something breaks and you get confused

import matplotlib
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches



class PrecisionCPD:
    def __init__(self, args):
        self.args = args
        self.optim_type = args.optim_type
        self.lam = args.lam
        self.M = args.M
        self.window_size = args.window_size
        self.post_window_size = args.post_window_size
        self.step_size = args.step_size
        self.full_basis = args.full_basis
        self.include_l1 = bool(args.include_l1)
        self.split_variance = bool(args.split_variance)
        self.iters = args.iters
        self.beta = args.beta
        self.t = args.t
        self.eps_matrices = args.eps_matrices
        self.num_eps_matrices = args.num_eps_mats
        self.fig_dir_path = args.fig_dir_path
        self.thresh_const = args.thresh_const

    # data assumed to be cleaned and normalized, passed in shape: [T, dim]
    def fit_glasso(self, data, use_thav=False):
        print("GLASSO DATA {}".format(data.shape))
        #lambda_search = [5e-2, 8e-2, 1e-1, 3e-1, 5e-1]
        C_train = np.cov(data.T)
        r_max = np.max(np.abs(C_train[np.triu_indices(C_train.shape[0], k=1)]))
        lambda_search = [0.05 + (j*(r_max-0.05))/40 for j in range(1, 41, 1)]
        if use_thav:
            precision, chosen_lamb = thav_gl_fn(data_train=data.T, lambda_search=lambda_search, C=0.7, threshold=1.0)
            print("Precision Shape Thav {} Chosen Lambda {}".format(precision.shape, chosen_lamb))
        else:     
            glasso = GraphicalLassoCV(alphas=lambda_search, n_refinements=4, tol=1e-4, max_iter=1500, cv=5).fit(data)
            precision = glasso.precision_
            chosen_lamb = glasso.alpha_
            tthresh = self.thresh_const*chosen_lamb
            print("Precision Shape Glasso {} Chosen Lambda {} Thresh {}".format(precision.shape, chosen_lamb, tthresh))
            precision[np.abs(precision) <= tthresh] = 0.0
        #precision[np.abs(precision) <= 0.1] = 0.0
        self.precision = precision

        #self.glasso = GraphicalLasso(max_iter=500, alpha=self.lam, tol=1e-5, verbose=False).fit(data)
        #print(np.count_nonzero(self.glasso.precision_)/len(self.glasso.precision_.flatten()))
        #exit()
        
        #self.inv_cov = inv(np.cov(data.T, bias=True))
        #self.inv_cov += np.eye(self.inv_cov.shape[0])*np.abs(np.linalg.eig(self.inv_cov)[0].min()) + 0.05
        #assert(is_pos_def(self.inv_cov))

    def top_down_search(self, Z, precision, dim, clust_dist_mat):
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
    
    def recursive_split(self, basis_mat):
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
        Z = linkage(pairwise_distances, method=self.args.linkage)
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
            if self.split_variance:
                if len(np.nonzero(A)[0]) > 0:
                    new_basis_matrices.append(vectorize_matrix(np.diag(np.diag(A.copy()))))
                np.fill_diagonal(A, 0)
            if len(np.nonzero(A)[0]) > 0: # checking for 0 entries
                new_basis_matrices.append(vectorize_matrix(A))
        new_basis_matrices = np.array(new_basis_matrices)
        if new_basis_matrices.shape[0] <= 1:
            print("...Recursion Complete for Dendrogram Criterion...")
            performed_split = False
            return new_basis_matrices, performed_split
        
        for i in range(new_basis_matrices.shape[0]):
            curr_mat = symmetrize_from_vector(new_basis_matrices[i], self.dim)
            nonzero_cols = np.nonzero(np.any(curr_mat != 0, axis=0))[0]
            #print(nonzero_cols)
            #print("NONZERO COLS {}".format(nonzero_cols))
            if len(nonzero_cols) < 2: # don't split into a singleton cluster - not informative - 3x3 is the minimum for splitting
                print("...Recursion Complete for Cluster Size {}...".format(len(nonzero_cols)))
                performed_split = False
                pass
            #print("Matrix {} Len {} Channels Contained {}".format(i, len(nonzero_cols), nonzero_cols))
            #print()
        #print("NEW BASIS MATRICES SHAPE", new_basis_matrices.shape)

        
        # if new_basis_matrices.shape[0] <= 1:
        #     performed_split = False

        return new_basis_matrices, performed_split



    def bfs_basis_mats(self, data_full, curr_basis_mats):
        #print("###################################### RECURSION BLOCK LEVEL ######################################")
        data_train = data_full[:, 0:int(self.args.train_percent*data_full.shape[1])]
        C_full = np.cov(data_train.copy(), bias=True)
        level_basis_mats = []
        g_prob = create_global_problem(curr_basis_mats, dim=self.dim)
        #alphas = solve_optim_global(curr_C=C_full, g_prob=g_prob)
        for i, mat in enumerate(curr_basis_mats):
            #print("TRAIN DATA {}".format(data_train.shape))
            orig_mat = mat.copy()
            mat = symmetrize_from_vector(mat, dim=self.dim)
            # if mat.shape[0] <= 4: # suitable localization - treat 4x4 or less as leaf nodes -> keep original matrix instead
            #     level_basis_mats.append(orig_mat)
            #     continue
            nonzero_cols = np.nonzero(np.any(mat != 0, axis=0))[0]
            if len(nonzero_cols) <= self.args.recursion_min:
                print("...Recursion Complete for Singleton Cluster...")
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
            anderson_lrt_value = self.anderson_lrt(cluster_precision=cluster_precision, C=train_C, N=data_train.shape[1])
            dof = 0.5*train_C.shape[0]*(train_C.shape[0]+1) - 1 # q here is just 1 since we are cluster specific
            chisquare_val = chi2.sf(anderson_lrt_value, dof)
            #print("CHISQUARE P-VAL {} DOF {}".format(chisquare_val, dof))
            new_mats, performed_split = self.recursive_split(mat)
            #print(new_mats.shape, performed_split)
            if performed_split:
                #print("BASIS MATS SHAPE {}".format(curr_basis_mats.shape))
                reduced_basis_mats = np.delete(curr_basis_mats, i, axis=0)
                #print("REDUCED MATS SHAPE {}".format(reduced_basis_mats.shape))
                updated_basis_matrices = np.concatenate((reduced_basis_mats, new_mats), axis=0)
                #print("UPDATED MATS SHAPE {}".format(updated_basis_matrices.shape))
                nonzero_cols_one = np.nonzero(np.any(symmetrize_from_vector(updated_basis_matrices[-2], self.dim) != 0, axis=0))[0]
                nonzero_cols_two = np.nonzero(np.any(symmetrize_from_vector(updated_basis_matrices[-1], self.dim) != 0, axis=0))[0]
                print("***********************\nTesting Split of {}\nInto {}\nand {}\n***********************".format(nonzero_cols, nonzero_cols_one, nonzero_cols_two))
                silhoutte_labels = self.cutree
                silhoutte_labels[nonzero_cols_two] = int(self.cutree.max() + 1)
                new_silhoutte_score = silhouette_score(self.root_dist_mat, silhoutte_labels, metric='precomputed')
                print("SILHOUETTE DIFFERENCE CURR {} NEW {}".format(self.curr_silhoutte_score, new_silhoutte_score))
                print("CHISQUARE VAL {}".format(chisquare_val))
                #print("Silhoutte Scores", new_silhoutte_score, self.curr_silhoutte_score)
                condition_one = len(nonzero_cols_one) >= self.args.recursion_min and len(nonzero_cols_two) >= self.args.recursion_min
                condition_two = new_silhoutte_score > self.curr_silhoutte_score
                condition_three = chisquare_val >= 1e-5
                

                if condition_one and (condition_two or condition_three): # we split and the scores improved or didn't degrade (p-value) -> take new 2 matrices!
                    #print("ADDING BASIS MATS")
                    #print("NEW MATS SHAPE", new_mats.shape)
                    self.curr_silhoutte_score = new_silhoutte_score
                    for matr in new_mats:
                        level_basis_mats.append(matr)
                else: # both of these are false -> keep original matrix instead
                    print("...Recursion Complete for Fit-Check and Silhouette Score...")
                    level_basis_mats.append(orig_mat)
            else: # split didn't take place -> keep original matrix instead
                level_basis_mats.append(orig_mat)
        #print(np.concatenate(level_basis_mats, axis=1).shape)
        level_basis_mats = np.array(level_basis_mats)
        if level_basis_mats.shape[0] > curr_basis_mats.shape[0]: # splits took place
            #print("$$$$$$$$$$$$$ RECURSING $$$$$$$$$$$$$$")
            #print("###################################### END OF RECURSION BLOCK LEVEL ######################################")
            level_basis_mats = self.bfs_basis_mats(data_full, level_basis_mats) # recurse down a level


        return level_basis_mats
            

    def construct_basis_matrices(self):
        #precision = self.glasso.precision_.copy()
        precision = self.precision.copy()
        #precision = self.inv_cov

        #####################
        zero_indices_prec = np.where(precision == 0.0)
        zero_indices_prec_tuples = [(zero_indices_prec[0][i], zero_indices_prec[1][i]) for i in range(len(zero_indices_prec[0]))]
        zero_indices_prec_tuples = list({*map(tuple, map(sorted, zero_indices_prec_tuples))})
        #####################

        self.dim = precision.shape[0]
        clust_dist_mat = np.abs(precision)
        np.fill_diagonal(clust_dist_mat, 0.0)
        ###########
        clust_dist_mat = (clust_dist_mat.max()+1e-5) - clust_dist_mat
        np.fill_diagonal(clust_dist_mat, 0.0)
        ###########
        #pairwise_distances = sch.distance.pdist(clust_dist_mat)
        pairwise_distances = squareform(clust_dist_mat)
        Z = linkage(pairwise_distances, method=self.args.linkage)
        ######
        # plot the dendrogram 
        if self.args.sap:
            plt.figure()
            dn = hierarchy.dendrogram(Z)
            #plt.savefig(os.path.join(self.fig_dir_path, "dendrogram.png"))
            plt.savefig(os.path.join('debugging_figs/sap_figs/', "dendrogram_{}.png".format(self.args.linkage)))
            plt.close()
            sns.heatmap(precision)
            plt.savefig(os.path.join('debugging_figs/sap_figs/', "precision_{}.png".format(self.args.linkage)))
            plt.close()
        else:
            plt.figure(figsize=(12,9))
            dn = hierarchy.dendrogram(Z)
            #plt.savefig(os.path.join(self.fig_dir_path, "dendrogram.png"))
            plt.yticks([])
            #plt.xlabel("Dimension Labels", fontsize=26)
            #plt.title("Sample Dendrogram", fontsize=28)
            plt.xlabel('Feature Index', fontsize=24)
            plt.savefig(os.path.join('debugging_figs/nonsap_figs/', "dendrogram_{}.png".format(self.args.linkage)))
            plt.close()
            sns.heatmap(precision)
            plt.savefig(os.path.join('debugging_figs/nonsap_figs/', "precision_{}.png".format(self.args.linkage)))
            plt.close()
        ######
        if self.args.recursion:
            print("*** Utilizing Recursion... ***")
            cutree1 = hierarchy.cut_tree(Z, n_clusters=self.args.base_M).squeeze() # start with just 2 or whatever the base is specified as
        else:
            print("*** NOT Utilizing Recursion... ***")
            cutree1 = hierarchy.cut_tree(Z, n_clusters=self.args.M).squeeze()
        root, nodelist = hierarchy.to_tree(Z, rd=True)
        #self.dendrogram = dn
        self.Z = Z
        self.cutree = cutree1
        self.root = root
        self.nodelist = nodelist
        self.root_dist_mat = clust_dist_mat
        #print(self.root)
        #print(self.nodelist)

        self.curr_silhoutte_score = silhouette_score(clust_dist_mat, cutree1, metric='precomputed')
        self.basis_matrices = []
        for i in range(max(set(cutree1))+1): # iterate over clusters
            idxs = np.where(cutree1 == i)[0] # indexes for given cluster
            A = np.zeros(precision.shape) # blank A matrix
            for idx in idxs: # loop over indexes
                for idx2 in idxs: # loop over indexes
                    A[idx][idx2] = precision[idx][idx2].copy() # set i,j entry to be the entry from precision matrix for given cluster
            if self.split_variance:
                if len(np.nonzero(A)[0]) > 0:
                    self.basis_matrices.append(vectorize_matrix(np.diag(np.diag(A.copy()))))
                np.fill_diagonal(A, 0)
            if len(np.nonzero(A)[0]) > 0:
                self.basis_matrices.append(vectorize_matrix(A))
        
        basis_mats_backup = np.array(self.basis_matrices).copy()
        #################
        if bool(self.eps_matrices):
            A = np.zeros(precision.shape)
            counter = 0
            for tup in zero_indices_prec_tuples:
                i,j = tup
                if counter < len(zero_indices_prec_tuples)//self.num_eps_matrices:
                    A[i,j] = 1e-1
                    A[j,i] = 1e-1
                    counter += 1
                else:
                    A[i,j] = 1e-1
                    A[j,i] = 1e-1
                    self.basis_matrices.append(vectorize_matrix(A.copy()))
                    A = np.zeros(precision.shape)
                    counter = 0
            if len(np.nonzero(A)[0]) > 0:
                self.basis_matrices.append(vectorize_matrix(A.copy()))
        #################

        self.basis_matrices = np.array(self.basis_matrices)


        #leftover_basis_matrix = precision - symmetrize_from_vector(self.basis_matrices.sum(axis=0), dim=self.dim)
        leftover_basis_matrix = precision - symmetrize_from_vector(basis_mats_backup.sum(axis=0), dim=self.dim)
        self.basis_matrices_full = np.concatenate((self.basis_matrices, np.expand_dims(vectorize_matrix(leftover_basis_matrix), 0)), 0)


        assert is_pos_def(symmetrize_from_vector(self.basis_matrices_full.sum(axis=0), dim=self.dim)), "Not PosDef"
        assert is_pos_def(symmetrize_from_vector(self.basis_matrices.sum(axis=0), dim=self.dim)), "Not PosDef"
        for mat in self.basis_matrices_full:
            assert is_symmetric(symmetrize_from_vector(mat, dim=self.dim)), "Not Symmetric"
        print("H Matrices:", self.basis_matrices.shape[0])
        #print("Sum {}".format((self.basis_matrices[0] == 0).sum()))
        #print("Sum {}".format((self.basis_matrices[1] == 0).sum()))
        print("Shape {}".format(self.basis_matrices[0].shape))
        #print("Shape {}".format(self.basis_matrices[1].shape))

    # # data_full assumed to be passed in shape: [dim, T]
    def perform_lrt_covariance(self, data_full):
        lrt_vals, p_vals, null_likelihoods, alt_likelihoods = LRT_covariance(data_full, window_size=self.window_size)

        return lrt_vals, p_vals

    # data_full assumed to be passed in shape: [dim, T]
    def perform_lrt_global(self, data_full):
        basis_mats = self.basis_matrices
        if bool(self.full_basis):
            basis_mats = self.basis_matrices_full
        lrt_vals, p_vals = LRT_all_coeffs_full_likelihood(data_full, M=basis_mats.shape[0], dim=data_full.shape[0], H_s=basis_mats, 
                                                          window_size=self.window_size, lam=self.lam, step_size=self.step_size, include_l1=self.include_l1, 
                                                          iters=self.iters, beta=self.beta)
        
        return np.array(lrt_vals), np.array(p_vals)
    
    def recursive_split_basis_matrix(self, basis_mats, greatest_change_mat_idx):
        greatest_change_mat = symmetrize_from_vector(basis_mats[greatest_change_mat_idx], dim=self.dim)
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
        Z = linkage(pairwise_distances, method=self.args.linkage)
        # BREAK INTO 2 NEW CLUSTERS
        cutree = hierarchy.cut_tree(Z, n_clusters=2).squeeze()
        #fclust_res = fcluster(Z, t=2, criterion='maxclust')
        ###################
        #cutree = fclust_res
        # new_silhoutte_score = silhouette_score(clust_dist_mat, cutree, metric='precomputed')
        ###################
        new_basis_matrices = []
        for i in range(min(set(cutree)), max(set(cutree))+1): # iterate over clusters
            idxs = np.where(cutree == i)[0] # indexes for given cluster
            A = np.zeros(greatest_change_mat.shape) # blank A matrix
            for idx in idxs: # loop over indexes
                for idx2 in idxs: # loop over indexes
                    first_idx = orig_nonzero_cols[idx] # remap back to original space
                    second_idx = orig_nonzero_cols[idx2] # remap back to original space
                    A[first_idx][second_idx] = greatest_change_mat[first_idx][second_idx].copy() # set i,j entry to be the entry from precision matrix for given cluster
                    #A[idx][idx2] = greatest_change_mat[idx][idx2].copy() # set i,j entry to be the entry from precision matrix for given cluster
            if self.split_variance:
                if len(np.nonzero(A)[0]) > 0:
                    new_basis_matrices.append(vectorize_matrix(np.diag(np.diag(A.copy()))))
                np.fill_diagonal(A, 0)
            if len(np.nonzero(A)[0]) > 0:
                new_basis_matrices.append(vectorize_matrix(A))
        new_basis_matrices = np.array(new_basis_matrices)
        for i in range(new_basis_matrices.shape[0]):
            curr_mat = symmetrize_from_vector(new_basis_matrices[i], self.dim)
            nonzero_cols = np.nonzero(np.any(curr_mat != 0, axis=0))[0]
            #print("Matrix {} Len {} Channels Contained {}".format(i, len(nonzero_cols), nonzero_cols))
            #print()
        #print("NEW BASIS MATRICES SHAPE", new_basis_matrices.shape)

        return new_basis_matrices

    def anderson_lrt(self, cluster_precision, C, N):
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
    
    def fit_optim_candidate_point(self, C_one, C_two, C_full, H_s, window_size, lam):
        coeffs_hat_total = optim_boyd(C=C_full, H_s=H_s)
        # null_likelihood = full_likelihood(coeffs_hat_total, H_s, C_full, N=window_size*2, 
        #                                   lam=lam, include_l1=False, debug_title='global')
        null_likelihood = full_likelihood(coeffs_hat_total, H_s, C_full, N=window_size+self.post_window_size, 
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
            alt_likelihood_alpha_i_pre = full_likelihood(alpha_i_change_pre, H_s, C_one, N=window_size, 
                                                         lam=lam, include_l1=False, debug_title='Pre')
            # likelihood on post data, alpha_one change
            alt_likelihood_alpha_i_post = full_likelihood(alpha_i_change_post, H_s, C_two, N=self.post_window_size, 
                                                          lam=lam, include_l1=False, debug_title='Post')
            alt_likelihood_alpha_i = alt_likelihood_alpha_i_pre + alt_likelihood_alpha_i_post
            #alt_likelihood_alpha_i_alt = alt_likelihood_alpha_i_pre_alt + alt_likelihood_alpha_i_post_alt
            #print("Likelihood Diff Total {}".format(alt_likelihood_alpha_i - alt_likelihood_alpha_i_alt))
            #dof = 0.5*C_full.shape[0]*(C_full.shape[0]+1) - (M + 1)
            dof = 2
            test_stat_i, p_val_i = likelihood_ratio_test(null_likelihood, 
                                                alt_likelihood_alpha_i, dof)
            test_stats_m.append(test_stat_i)
            p_vals_m.append(p_val_i)
        
        return np.expand_dims(np.array(test_stats_m), 0), meinshausen_correction(H_s=H_s, p_vals_all=np.expand_dims(np.array(p_vals_m), 0), dim=self.dim, log_pvals=self.args.log_pvals)
    
    def recurse_on_candidate_point(self, lrt_vals_all, p_vals_corrected, data_full, basis_mats, candidate_point):
        """
        Input is p-vals from current level of recursion AT candidate point
        Shape: (1, M)

        This function recurses down a level
        """
        greatest_change_mat_idx = p_vals_corrected[0, :].argmin() # this should work if it's 1-d just expanded
        greatest_change_mat = symmetrize_from_vector(basis_mats[greatest_change_mat_idx], dim=self.dim)
        nonzero_cols = np.nonzero(np.any(greatest_change_mat != 0, axis=0))[0]
        #print(data_full.shape)
        print("GREATEST CHANGE MATRIX CHANNELS CONTAINED {}".format(nonzero_cols))
        #print(p_vals_corrected.shape)

        # store data for recursion - cp identification
        data_one = data_full[:, 0:self.args.window_size]
        data_two = data_full[:, candidate_point:(candidate_point + self.post_window_size)]
        data_total_window = np.concatenate((data_one, data_two), axis=1)
        C_one = np.cov(data_one, bias=True)
        C_one = C_one + np.eye(C_one.shape[0])*1e-8
        C_two = np.cov(data_two, bias=True)
        C_two = C_two + np.eye(C_two.shape[0])*1e-8 # correct numerical instability
        C_total = np.cov(data_total_window, bias=True)
        C_total = C_total + np.eye(C_total.shape[0])*1e-8
        
        # store training data for model fitting - goodness of fit check
        data_train = data_full[:, 0:int(self.args.train_percent*data_full.shape[1])]
        #print("TRAIN DATA {}".format(data_train.shape))
        C_full = np.cov(data_train.copy(), bias=True)
        data_train = data_train[nonzero_cols, :]
        if len(nonzero_cols) < self.args.recursion_min: # rerun it and stop
            lrt_vals_all, p_vals_all = LRT_individual_coeffs_full_likelihood(data_full, M=basis_mats.shape[0], dim=data_full.shape[0], H_s=basis_mats, 
                                                                         window_size=self.window_size, post_window_size=self.post_window_size, lam=self.lam, step_size=self.step_size, include_l1=self.include_l1, 
                                                                         iters=self.iters, beta=self.beta, t=self.t, optim_type=self.optim_type, args=self.args)
            
            lrt_vals_all = np.array(lrt_vals_all)
            p_vals_all = np.array(p_vals_all)
            #p_vals_corrected = np.array(apply_bonferroni_correction(p_vals_all))
            #p_vals_corrected = np.array(apply_fdr_correction(p_vals_all))
            p_vals_corrected = meinshausen_correction(basis_mats, p_vals_all, dim=data_full.shape[0], log_pvals=self.args.log_pvals)

            return lrt_vals_all, p_vals_corrected
        train_C = np.cov(data_train.copy(), bias=True)
        train_C = train_C + np.eye(train_C.shape[0])*1e-8

        # fit current level of recursion on training data
        alphas = optim_boyd(C=C_full, H_s=basis_mats)
        #print("ALPHAS BOYD {}".format(alphas))
        #print("Train C Shape {}".format(train_C.shape))
        #print("Alphas {}".format(alphas))
        cluster_precision = greatest_change_mat[~np.all(greatest_change_mat == 0, axis=1)]
        cluster_precision = alphas[greatest_change_mat_idx]*cluster_precision[:, ~np.all(cluster_precision == 0, axis=0)]

        anderson_lrt_value = self.anderson_lrt(cluster_precision=cluster_precision, C=train_C, N=data_train.shape[1])
        dof = 0.5*train_C.shape[0]*(train_C.shape[0]+1) - 1 # q here is just 1 since we are cluster specific
        chisquare_val = chi2.sf(anderson_lrt_value, dof)

        cluster_eigvals = sorted(np.linalg.eig(cluster_precision)[0], reverse=True)
        cluster_eigval_max = max(cluster_eigvals)
        cluster_eigval_min = min(cluster_eigvals)
        eigval_ratio = cluster_eigval_max/cluster_eigval_min # condition number
        print("Eigvals", cluster_eigvals)
        print("Eigval ratio", eigval_ratio)
        #print("CHISQUARE P-VAL {} DOF {}".format(chisquare_val, dof))
        # if conditions are met, recurse
        if len(nonzero_cols) >= self.args.recursion_min and eigval_ratio >= self.args.condition_number_thresh: # first stopping conditions
            new_basis_matrices = self.recursive_split_basis_matrix(basis_mats, greatest_change_mat_idx)
            #print("NUMBER OF NEW BASIS MATRICES {}".format(new_basis_matrices.shape[0]))
            nonzero_cols_one = np.nonzero(np.any(symmetrize_from_vector(new_basis_matrices[0], self.dim) != 0, axis=0))[0]
            if new_basis_matrices.shape[0] > 1:
                #print("RECALCULATING SILHOUETTE SCORE")
                #print("CURR CUTREE", self.cutree)
                nonzero_cols_two = np.nonzero(np.any(symmetrize_from_vector(new_basis_matrices[1], self.dim) != 0, axis=0))[0]
                #print("NONZERO COLS", nonzero_cols_two)
                if len(nonzero_cols_one) >= self.args.recursion_min or len(nonzero_cols_two) >= self.args.recursion_min: # if the clustering is able to be split, recurse
                    self.cutree[nonzero_cols_two] = int(self.cutree.max() + 1)
                    #print("NEW CUTREE", self.cutree)
                    new_silhoutte_score = silhouette_score(self.root_dist_mat, self.cutree, metric='precomputed')
                    #print("Silhoutte Scores", new_silhoutte_score, self.curr_silhoutte_score)
                    reduced_basis_mats = np.delete(basis_mats, greatest_change_mat_idx, axis=0)
                    updated_basis_matrices = np.concatenate((reduced_basis_mats, new_basis_matrices), axis=0)
                    self.basis_matrices = updated_basis_matrices
                    condition_one = new_basis_matrices.shape[0] > 1
                    condition_two = new_silhoutte_score > self.curr_silhoutte_score
                    condition_three = chisquare_val >= 1e-5
                    #print("Condition One {} Two {} Three {}".format(condition_one, condition_two, condition_three))
                    #print("\n\n############# RECURSING #################\n")
                    self.curr_silhoutte_score = new_silhoutte_score
                    _, p_vals_corrected = self.fit_optim_candidate_point(C_one=C_one, C_two=C_two, C_full=C_total, H_s=basis_mats, window_size=self.window_size, lam=self.lam)
                    return self.recurse_on_candidate_point(lrt_vals_all, p_vals_corrected=p_vals_corrected, data_full=data_full, basis_mats=self.basis_matrices, candidate_point=candidate_point)
                else:
                    print("\n\n************** NOT RECURSING - TOO SMALL OF SPLIT {} {}**************\n".format(len(nonzero_cols_one), len(nonzero_cols_two)))
            else:
                print("\n\n************** NOT RECURSING - BASIS MATRIX SHAPE **************\n")


        lrt_vals_all, p_vals_all = LRT_individual_coeffs_full_likelihood(data_full, M=basis_mats.shape[0], dim=data_full.shape[0], H_s=basis_mats, 
                                                                         window_size=self.window_size, post_window_size=self.post_window_size, lam=self.lam, step_size=self.step_size, include_l1=self.include_l1, 
                                                                         iters=self.iters, beta=self.beta, t=self.t, optim_type=self.optim_type, args=self.args)
        
        lrt_vals_all = np.array(lrt_vals_all)
        p_vals_all = np.array(p_vals_all)
        #p_vals_corrected = np.array(apply_bonferroni_correction(p_vals_all))
        #p_vals_corrected = np.array(apply_fdr_correction(p_vals_all))
        p_vals_corrected = meinshausen_correction(basis_mats, p_vals_all, dim=data_full.shape[0], log_pvals=self.args.log_pvals)

        return lrt_vals_all, p_vals_corrected




    def perform_lrt_local(self, data_full):
        basis_mats = self.basis_matrices
        """
        ADD BFS RECURSION HERE
        """
        if self.args.recursion:
            print("*** -----> TOP DOWN SEARCH <----- ***")
            basis_mats, cutree1 = self.top_down_search(Z=self.Z, precision=self.precision, dim=self.dim, clust_dist_mat=self.root_dist_mat)
            #basis_mats = self.bfs_basis_mats(data_full, basis_mats)
            self.basis_matrices = basis_mats
        """
        """
        self.print_clusters_rv()
        # if bool(self.full_basis):
        #     basis_mats = self.basis_matrices_full
        lrt_vals_all, p_vals_all = LRT_individual_coeffs_full_likelihood(data_full, M=basis_mats.shape[0], dim=data_full.shape[0], H_s=basis_mats, 
                                                                         window_size=self.window_size, post_window_size=self.post_window_size, lam=self.lam, step_size=self.step_size, include_l1=self.include_l1, 
                                                                         iters=self.iters, beta=self.beta, t=self.t, optim_type=self.optim_type, args=self.args)

        #return np.array(lrt_vals_all), np.array(apply_fdr_correction(p_vals_all))
        lrt_vals_all = np.array(lrt_vals_all)
        p_vals_all = np.array(p_vals_all)
        #p_vals_corrected = np.array(apply_bonferroni_correction(p_vals_all))
        #p_vals_corrected = np.array(apply_fdr_correction(p_vals_all))
        p_vals_corrected = meinshausen_correction(basis_mats, p_vals_all, dim=data_full.shape[0], log_pvals=self.args.log_pvals)
        candidate_cp = p_vals_corrected.min(axis=1).argmin()

        

        """
        LIKELIHOOD RATIO TEST FROM ANDERSON 1970 FOR RECURSION BASE CASE/SPLITTING CONDITION
        
        1) CHECK GREATEST CHANGE MAT SIZE > 2
        2) CHECK LRT P VALUE < SOME CUTOFF
        DO SOME SPLITTING WHILE ABOVE CONDITIONS OR WHATEVER
        FIN
        """
        
        """
        RECURSION IN PROGRESS HERE IN COMMENTS IS DFS RECURSION
        """
        if self.args.candidate_recursion:
            print("**************PRE RECURSION MATRICES**************")
            self.print_clusters_rv()
            print("Number of Basis Matrices {}".format(basis_mats.shape[0]))
            print("**************************************************")
            print("***********************")
            print("Recursing on Candidate Point...")
            lrt_vals_all, p_vals_corrected = self.recurse_on_candidate_point(lrt_vals_all, np.expand_dims(p_vals_corrected[candidate_cp, :], 0), data_full, basis_mats, candidate_cp)
            print("***********************")
        """
        """
        # greatest_change_mat_idx = p_vals_corrected[candidate_cp, :].argmin()
        # greatest_change_mat = symmetrize_from_vector(basis_mats[greatest_change_mat_idx], dim=self.dim)
        # nonzero_cols = np.nonzero(np.any(greatest_change_mat != 0, axis=0))[0]
        # print(data_full.shape)
        # print("GREATEST CHANGE MATRIX CHANNELS CONTAINED {}".format(nonzero_cols))

        # # store data for recursion - cp identification
        # data_one = data_full[:, 0:self.args.window_size]
        # data_two = data_full[:, candidate_cp:(candidate_cp + self.args.window_size)]
        # data_total_window = np.concatenate((data_one, data_two), axis=1)

        # # store training data for model fitting - goodness of fit check
        # data_train = data_full[:, 0:int(self.args.train_percent*data_full.shape[1])]
        # print("TRAIN DATA {}".format(data_train.shape))
        # C_full = np.cov(data_train.copy(), bias=True)
        # data_train = data_train[nonzero_cols, :]
        # train_C = np.cov(data_train.copy(), bias=True)
        # train_C = train_C + np.eye(train_C.shape[0])*1e-7
        
        # alphas = optim_boyd(C=C_full, H_s=basis_mats)
        # print("ALPHAS BOYD {}".format(alphas))
        # print("Train C Shape {}".format(train_C.shape))
        # #print("Alphas {}".format(alphas))
        # cluster_precision = greatest_change_mat[~np.all(greatest_change_mat == 0, axis=1)]
        # cluster_precision = alphas[greatest_change_mat_idx]*cluster_precision[:, ~np.all(cluster_precision == 0, axis=0)]
        
        # anderson_lrt_value = self.anderson_lrt(cluster_precision=cluster_precision, C=train_C, N=data_train.shape[1])
        # dof = 0.5*train_C.shape[0]*(train_C.shape[0]+1) - 1 # q here is just 1 since we are cluster specific
        # chisquare_val = chi2.sf(anderson_lrt_value, dof)
        # print("CHISQUARE P-VAL {} DOF {}".format(chisquare_val, dof))
        # # if conditions are met, recurse
        # if len(nonzero_cols) > 2: # first stopping conditions
        #     new_basis_matrices = self.recursive_split_basis_matrix(basis_mats, p_vals_corrected)
        #     nonzero_cols_one = np.nonzero(np.any(symmetrize_from_vector(new_basis_matrices[0], self.dim) != 0, axis=0))[0]
        #     if new_basis_matrices.shape[0] > 1:
        #         print("RECALCULATING SILHOUETTE SCORE")
        #         print("CURR CUTREE", self.cutree)
        #         nonzero_cols_two = np.nonzero(np.any(symmetrize_from_vector(new_basis_matrices[1], self.dim) != 0, axis=0))[0]
        #         print("NONZERO COLS", nonzero_cols_two)
        #         self.cutree[nonzero_cols_two] = int(self.cutree.max() + 1)
        #         print("NEW CUTREE", self.cutree)
        #     new_silhoutte_score = silhouette_score(self.root_dist_mat, self.cutree, metric='precomputed)
        #     print("Silhoutte Scores", new_silhoutte_score, self.curr_silhoutte_score)
        #     reduced_basis_mats = np.delete(basis_mats, greatest_change_mat_idx, axis=0)
        #     updated_basis_matrices = np.concatenate((reduced_basis_mats, new_basis_matrices), axis=0)
        #     self.basis_matrices = updated_basis_matrices
        #     if new_basis_matrices.shape[0] > 1 and (new_silhoutte_score > self.curr_silhoutte_score or chisquare_val >= 1e-5): # if the clustering is able to be split, recurse
        #         self.curr_silhoutte_score = new_silhoutte_score
        #         return self.perform_lrt_local(data_full=data_full)
        """
        END RECURSION
        """

        #p_vals_corrected = p_vals_all
        #return np.array(lrt_vals_all), np.array(apply_bonferroni_correction(p_vals_all))
        #print(p_vals_all)
        #print("P VALS {}", p_vals_corrected)
        return lrt_vals_all, p_vals_corrected

    def print_clusters_rv(self):
        basis_mats = self.basis_matrices
        if bool(self.full_basis):
            basis_mats = self.basis_matrices_full
        m = np.zeros((self.dim, self.dim))
        for i in range(basis_mats.shape[0]):
            curr_mat = symmetrize_from_vector(basis_mats[i], self.dim)
            nonzero_cols = np.nonzero(np.any(curr_mat != 0, axis=0))[0]
            for col in nonzero_cols:
                for col2 in nonzero_cols:
                    m[col, col2] = i+1
            print("****************************************")
            if i == (basis_mats.shape[0] - 1):
                print("Basis Matrix {}".format(i))
            elif self.split_variance and i == (basis_mats.shape[0] - 2):
                print("Variance Basis Matrix {}".format(i))
            else:
                print("Basis Matrix {}".format(i))
            print("Channels Contained {}".format(nonzero_cols))
            print("****************************************")
            print()
        #print(m)
        #colors = 'blue lime yellow magenta red'.split()
        # colors = 'whitesmoke blue lime cyan red orange'.split()
        # values = range(basis_mats.shape[0]+1)
        # #patches = [mpatches.Patch(color=colors[x], label="Cluster {l}".format(l=values[x]) ) for x in range(len(values)) ]
        # cmap = matplotlib.colors.ListedColormap(colors, name='colors', N=None)

        # plt.imshow(m, cmap=cmap)
        # plt.xticks(range(0, 20), rotation=45)
        # plt.yticks(range(0, 20), rotation=45)
        # #plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=-1.0 )
        # plt.grid(True)
        # plt.savefig(os.path.join('debugging_figs/nonsap_figs/', "precision_{}.png".format(self.args.linkage)))
        # plt.close()
    
    def save_matrices_simulations(self, save_path):
        print("Saving Basis Mats")
        basis_mats = self.basis_matrices
        if bool(self.full_basis):
            basis_mats = self.basis_matrices_full
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        for i in range(basis_mats.shape[0]):
            np.savetxt(os.path.join(save_path, "matrix_{}.csv".format(i)), basis_mats[i], delimiter=',')

    def permute_blocks(self):
        self.permuted_mats = []
        self.min_eigvals = []
        for i,mat in enumerate(self.basis_matrices):
            curr_mat = symmetrize_from_vector(mat, self.dim)
            zero_cols = np.where(~curr_mat.any(axis=0))[0]
            new_mat = np.delete(curr_mat, zero_cols, axis=0)
            new_mat = np.delete(new_mat, zero_cols, axis=1)
            #print(new_mat.shape)
            #print(new_mat)
            #sns.heatmap(new_mat)
            #plt.show()
            self.permuted_mats.append(new_mat)
            min_eigval_curr = np.min(np.linalg.eig(new_mat)[0])
            self.min_eigvals.append(min_eigval_curr)
            print(np.linalg.eig(new_mat)[0])
            print(min_eigval_curr)
        self.min_eigvals = np.array(self.min_eigvals)
        
        

    



    



    