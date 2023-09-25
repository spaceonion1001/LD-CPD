import numpy as np
from numpy.linalg import inv as inv
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.extmath import fast_logdet
from optim import optim_boyd, unbiased_init_precision, solve_optim_global, create_global_problem
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


from statsmodels.tsa.seasonal import STL

from utils import is_symmetric, is_pos_def, vectorize_matrix, symmetrize_from_vector
from likelihood import LRT_all_coeffs, LRT_all_coeffs_full_likelihood, LRT_individual_coeffs_full_likelihood, apply_fdr_correction, LRT_covariance
from likelihood import apply_bonferroni_correction
from meinshausen import meinshausen_correction
import warnings
warnings.filterwarnings('ignore')  # <- remember to comment this if something breaks and you get confused


class PrecisionCPD:
    def __init__(self, args):
        self.args = args
        self.optim_type = args.optim_type
        self.lam = args.lam
        self.M = args.M
        self.window_size = args.window_size
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

    # data assumed to be cleaned and normalized, passed in shape: [T, dim]
    def fit_glasso(self, data):
        print("GLASSO DATA {}".format(data.shape))
        self.glasso = GraphicalLasso(max_iter=100, alpha=self.lam, tol=1e-5, verbose=False).fit(data)
        #self.inv_cov = inv(np.cov(data.T, bias=True))
        #self.inv_cov += np.eye(self.inv_cov.shape[0])*np.abs(np.linalg.eig(self.inv_cov)[0].min()) + 0.05
        #assert(is_pos_def(self.inv_cov))

    def construct_basis_matrices(self):
        precision = self.glasso.precision_.copy()
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
        pairwise_distances = sch.distance.pdist(clust_dist_mat)
        #pairwise_distances = squareform(clust_dist_mat)
        Z = linkage(pairwise_distances, method='average')
        ######
        # plot the dendrogram 
        # plt.figure()
        #dn = hierarchy.dendrogram(Z)
        #plt.savefig(os.path.join(self.fig_dir_path, "dendrogram.png"))
        #plt.close()
        ######
        cutree1 = hierarchy.cut_tree(Z, n_clusters=self.M).squeeze()
        root, nodelist = hierarchy.to_tree(Z, rd=True)
        #self.dendrogram = dn
        self.Z = Z
        self.cutree = cutree1
        self.root = root
        self.nodelist = nodelist
        
        #print(self.root)
        #print(self.nodelist)
        
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
        print("Shape {}".format(self.basis_matrices[1].shape))

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
    
    def recursive_split_basis_matrix(self, basis_mats, p_vals_corrected):
        candidate_cp = p_vals_corrected.min(axis=1).argmin()
        greatest_change_mat_idx = p_vals_corrected[candidate_cp, :].argmin()
        greatest_change_mat = symmetrize_from_vector(basis_mats[greatest_change_mat_idx], dim=self.dim)
        print("GREATEST CHANGE IDX: ", greatest_change_mat_idx, greatest_change_mat.shape)
        # RECLUSTER - SIMPLEST SOLUTION CURRENTLY
        clust_dist_mat = np.abs(greatest_change_mat)
        np.fill_diagonal(clust_dist_mat, 0.0)
        ###########
        clust_dist_mat = (clust_dist_mat.max()+1e-5) - clust_dist_mat
        np.fill_diagonal(clust_dist_mat, 0.0)
        ###########
        pairwise_distances = sch.distance.pdist(clust_dist_mat)
        Z = linkage(pairwise_distances, method='average')
        # BREAK INTO 2 NEW CLUSTERS
        cutree = hierarchy.cut_tree(Z, n_clusters=2).squeeze()
        fclust_res = fcluster(Z, t=2, criterion='maxclust')
        ###################
        cutree = fclust_res
        ###################

        new_basis_matrices = []
        for i in range(min(set(cutree)), max(set(cutree))+1): # iterate over clusters
            idxs = np.where(cutree == i)[0] # indexes for given cluster
            A = np.zeros(greatest_change_mat.shape) # blank A matrix
            for idx in idxs: # loop over indexes
                for idx2 in idxs: # loop over indexes
                    A[idx][idx2] = greatest_change_mat[idx][idx2].copy() # set i,j entry to be the entry from precision matrix for given cluster
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
            print("Matrix {} Len {} Channels Contained {}".format(i, len(nonzero_cols), nonzero_cols))
            print()
        print(new_basis_matrices.shape)

        return new_basis_matrices

    def anderson_lrt(self, cluster_precision, C, N):
        """
        Anderson LRT for goodness-of-fit

        -2*log of 4.10

        -Nlogdet(C)-Nlogdet(precision)
        """
        
        # calculate likelihood ratio criterion eq 4.10
        print("C {} Prec {}".format(C.shape, cluster_precision.shape))
        first_term = 0.5*N*fast_logdet(C)
        second_term = 0.5*N*fast_logdet(cluster_precision)
        res = -2*(first_term + second_term)
        print("First Term {} Second Term {} Res {}".format(first_term, second_term, res))

        return res


    def perform_lrt_local(self, data_full):
        basis_mats = self.basis_matrices
        print(basis_mats.shape[0])
        if bool(self.full_basis):
            basis_mats = self.basis_matrices_full
        lrt_vals_all, p_vals_all = LRT_individual_coeffs_full_likelihood(data_full, M=basis_mats.shape[0], dim=data_full.shape[0], H_s=basis_mats, 
                                                                         window_size=self.window_size, lam=self.lam, step_size=self.step_size, include_l1=self.include_l1, 
                                                                         iters=self.iters, beta=self.beta, t=self.t, optim_type=self.optim_type)

        #return np.array(lrt_vals_all), np.array(apply_fdr_correction(p_vals_all))
        lrt_vals_all = np.array(lrt_vals_all)
        p_vals_all = np.array(p_vals_all)
        #p_vals_corrected = np.array(apply_bonferroni_correction(p_vals_all))
        #p_vals_corrected = np.array(apply_fdr_correction(p_vals_all))
        p_vals_corrected = meinshausen_correction(basis_mats, p_vals_all, dim=data_full.shape[0])
        

        """
        TODO
        LIKELIHOOD RATIO TEST FROM ANDERSON 1970 FOR RECURSION BASE CASE/SPLITTING CONDITION
        
        1) CHECK GREATEST CHANGE MAT SIZE > 2
        2) CHECK LRT P VALUE < SOME CUTOFF
        DO SOME SPLITTING WHILE ABOVE CONDITIONS OR WHATEVER
        FIN
        """
        
        """
        RECURSION IN PROGRESS
        """
        candidate_cp = p_vals_corrected.min(axis=1).argmin()
        greatest_change_mat_idx = p_vals_corrected[candidate_cp, :].argmin()
        greatest_change_mat = symmetrize_from_vector(basis_mats[greatest_change_mat_idx], dim=self.dim)
        nonzero_cols = np.nonzero(np.any(greatest_change_mat != 0, axis=0))[0]
        print(data_full.shape)
        print("GREATEST CHANGE MATRIX CHANNELS CONTAINED {}".format(nonzero_cols))

        data_train = data_full[:, 0:int(self.args.train_percent*data_full.shape[1])]
        print("TRAIN DATA {}".format(data_train.shape))
        C_full = np.cov(data_train, bias=True)
        data_train = data_train[nonzero_cols, :]
        train_C = np.cov(data_train, bias=True)
        train_C = train_C + np.eye(train_C.shape[0])*1e-7
        
        g_prob = create_global_problem(basis_mats, dim=self.dim)
        alphas = optim_boyd(C=C_full, H_s=basis_mats)
        print("ALPHAS BOYD {}".format(alphas))
        #alphas = solve_optim_global(curr_C=C_full, g_prob=g_prob)
        #print("ALPHAS CVX {}".format(alphas))
        #alphas = unbiased_init_precision(C=C_full, H_s=basis_mats)
        #print("ALPHAS UNBIASED {}".format(alphas))
        
        print("Train C Shape {}".format(train_C.shape))
        #print("Alphas {}".format(alphas))
        cluster_precision = greatest_change_mat[~np.all(greatest_change_mat == 0, axis=1)]
        cluster_precision = alphas[greatest_change_mat_idx]*cluster_precision[:, ~np.all(cluster_precision == 0, axis=0)]
        
        anderson_lrt_value = self.anderson_lrt(cluster_precision=cluster_precision, C=train_C, N=data_train.shape[1])
        dof = 0.5*train_C.shape[0]*(train_C.shape[0]+1) - 1 # q here is just 1 since we are cluster specific
        chisquare_val = chi2.sf(anderson_lrt_value, dof)
        print("CHISQUARE P-VAL {} DOF {}".format(chisquare_val, dof))
        # if conditions are met, recurse
        if len(nonzero_cols) > 2 and chisquare_val >= 1e-5:
            new_basis_matrices = self.recursive_split_basis_matrix(basis_mats, p_vals_corrected)
            reduced_basis_mats = np.delete(basis_mats, greatest_change_mat_idx, axis=0)
            updated_basis_matrices = np.concatenate((reduced_basis_mats, new_basis_matrices), axis=0)
            self.basis_matrices = updated_basis_matrices
            if new_basis_matrices.shape[0] > 1: # if the clustering won't go down a level
                return self.perform_lrt_local(data_full=data_full)
        """
        END RECURSION
        """

        #p_vals_corrected = p_vals_all
        #return np.array(lrt_vals_all), np.array(apply_bonferroni_correction(p_vals_all))
        #print(p_vals_all)
        #print(p_vals_corrected)
        return lrt_vals_all, p_vals_corrected

    def print_clusters_rv(self):
        basis_mats = self.basis_matrices
        if bool(self.full_basis):
            basis_mats = self.basis_matrices_full
        for i in range(basis_mats.shape[0]):
            curr_mat = symmetrize_from_vector(basis_mats[i], self.dim)
            nonzero_cols = np.nonzero(np.any(curr_mat != 0, axis=0))[0]
            print("****************************************")
            if i == (basis_mats.shape[0] - 1):
                print("Leftover Basis Matrix {}".format(i))
            elif self.split_variance and i == (basis_mats.shape[0] - 2):
                print("Variance Basis Matrix {}".format(i))
            else:
                print("Basis Matrix {}".format(i))
            print("Channels Contained {}".format(nonzero_cols))
            print("****************************************")
            print()
    
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
        
        

    



    



    