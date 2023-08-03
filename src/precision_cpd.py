import numpy as np
from numpy.linalg import inv as inv
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
#from torch import zero_
sns.set()

from sklearn.covariance import graphical_lasso, GraphicalLasso, GraphicalLassoCV
from tqdm import tqdm

from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.cluster.hierarchy as sch


from statsmodels.tsa.seasonal import STL

from utils import is_symmetric, is_pos_def, vectorize_matrix, symmetrize_from_vector
from likelihood import LRT_all_coeffs, LRT_all_coeffs_full_likelihood, LRT_individual_coeffs_full_likelihood, apply_fdr_correction, LRT_covariance
from likelihood import apply_bonferroni_correction
from meinshausen import meinshausen_correction
import warnings
warnings.filterwarnings('ignore')  # <- remember to comment this if something breaks and you get confused


class PrecisionCPD:
    def __init__(self, args):
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
        Z = linkage(pairwise_distances, method='average')
        ######
        # plot the dendrogram 
        # plt.figure()
        # dn = hierarchy.dendrogram(Z)
        # plt.savefig(os.path.join(self.fig_dir_path, "dendrogram.png"))
        # plt.close()
        ######
        cutree1 = hierarchy.cut_tree(Z, n_clusters=self.M).squeeze()
        #self.dendrogram = dn
        self.Z = Z
        self.cutree = cutree1


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
        
        

    



    



    