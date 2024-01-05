from re import L
from venv import create
import numpy as np
from sklearn.datasets import make_spd_matrix, make_sparse_spd_matrix
from sklearn.preprocessing import scale
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV
from utils import is_pos_def, is_symmetric, symmetrize_from_vector, symmetrize_from_vector_alt, scale_data
from numpy.linalg import inv as inv
from statsmodels.tsa.vector_ar.var_model import VARProcess
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import os

from utils import vectorize_matrix, symmetrize_from_vector
import networkx as nx
import copy

def gilbert_graph(dim, seed):
    gr = nx.fast_gnp_random_graph(n=dim, p=3/dim, seed=seed, directed=False)
    A = nx.adjacency_matrix(gr).todense()
    #np.fill_diagonal(A, 1)
    return A

def lacz_sampling(adj):
    """
    adj: adjacency matrix
    """
    A = adj.astype(float)
    dim = A.shape[0]
    for j in range(1, dim):
        for i in range(j):
            if A[i, j] == 1:
                val_neg = np.random.uniform(-0.9, -0.5)
                val_pos = np.random.uniform(0.5, 0.9)
                pos_neg = np.random.choice([-1, 1])
                if pos_neg < 0:
                    chosen_val = val_neg
                else:
                    chosen_val = val_pos
                A[i][j] = chosen_val
                A[j][i] = A[i][j]
    min_eigval = -np.round(np.linalg.eig(A)[0].min(), decimals=1) + 0.05
    A = A + min_eigval*np.eye(dim)
    A = 1/(min_eigval) * A

    return A


def generate_matrices_orthogonal(prec_coeffs=None, M=2, dim=4, to_print=True, lam=1e-1, seed=42, linkage_type='single'):
    print(">>  Generating H Matrices - M {}; Dim {}; Lambda {}  <<".format(M, dim, lam))
    H_s = []
    if not prec_coeffs:
        prec_coeffs = np.random.rand(M)
    if (prec_coeffs <= 0.).sum() > 0:
        prec_coeffs = prec_coeffs + np.abs(prec_coeffs.min()) + 0.5 # add adjustment so non-negative
    """"""
    #precision = make_spd_matrix(dim)
    #precision = make_sparse_spd_matrix(dim=dim, alpha=0.9, norm_diag=True, random_state=seed)
    #delta = np.abs(np.min(np.linalg.eig(precision)[0])) + 0.01
    #delta_times_I = np.eye(dim)*delta
    #precision = precision + delta_times_I
    """"""

    adj = gilbert_graph(dim=dim, seed=seed)
    precision = lacz_sampling(adj)



    #print((precision == 0).sum()/len(precision.flatten()))
    #####
    # data_temp, _ = sim_data(covar=inv(precision), dim=dim, N=20000)
    # #glasso = GraphicalLasso(max_iter=1500, alpha=0.05, tol=1e-4, verbose=False).fit(data_temp.T)
    # glasso = GraphicalLassoCV(alphas=[1e-1, 1e-2, 5e-1, 5e-2, 1.0], n_refinements=4, tol=1e-4, max_iter=1500, cv=5).fit(data_temp.T)
    # precision = glasso.precision_.copy()
    ####

    # precision = (precision+precision.T)/2
    # precision = (precision - precision.min())/(precision.max() - precision.min())*2 - 1
    # eps_mask = (np.abs(precision) <= 0.2)
    # precision[eps_mask] = 0.0
    # np.fill_diagonal(precision, 1.0)
    # precision = get_near_psd(precision)
    # precision = precision/precision.max()
    # print(np.linalg.eig(precision)[0].max())
    ####
    
    #print(precision.max(), precision.min(), np.diag(precision))
    assert is_pos_def(precision), "Not Positive Definite Precision Matrix"
    clust_dist_mat = np.abs(precision)
    np.fill_diagonal(clust_dist_mat, 0.0)
    clust_dist_mat = (clust_dist_mat.max()+1e-5) - clust_dist_mat
    np.fill_diagonal(clust_dist_mat, 0.0)
    pairwise_distances = hierarchy.distance.pdist(clust_dist_mat)
    Z = linkage(pairwise_distances, method=linkage_type)
    cutree1 = hierarchy.cut_tree(Z, n_clusters=M).squeeze()
    for i in range(max(set(cutree1))+1): # iterate over clusters
        idxs = np.where(cutree1 == i)[0] # indexes for given cluster
        A = np.zeros(precision.shape) # blank A matrix
        for idx in idxs: # loop over indexes
            for idx2 in idxs: # loop over indexes
                A[idx][idx2] = precision[idx][idx2].copy() # set i,j entry to be the entry from precision matrix for given cluster
        if len(np.nonzero(A)[0]) > 0:
            H_s.append(vectorize_matrix(A))
    H_s = np.array(H_s)
    ##############
    # precision = np.zeros((dim, dim))
    # H_s_stacked = np.zeros((dim**2, M))
    # mat = make_spd_matrix(dim)
    # indxes = np.random.choice(np.arange(dim), (M, dim//M), replace=False)
    # for i in range(M):
    #     A = np.zeros((dim, dim))
    #     for idx in indxes[i]:
    #         for idx2 in indxes[i]:
    #             A[idx, idx2] = mat[idx, idx2]
    #     H_s.append(A)
    #     precision += prec_coeffs[i]*A
    #     H_s_stacked[:, i] = A.flatten()
    # H_s = np.array(H_s)
    
    # # ensure basis matrices are linearly independent
    # assert np.linalg.matrix_rank(H_s_stacked) == M, "Not Linearly Independent basis matrices"
    
    # # ensure it's actually symmetric
    # # minimal modification on scale of 1e-15
    # precision_corrected = (precision + precision.T)/2
    # assert is_pos_def(precision_corrected), "Not Positive Definite Initial Matrices"
    ##############
    if to_print:
        for i in range(H_s.shape[0]):
            curr_mat = symmetrize_from_vector(H_s[i], dim)
            #curr_mat = H_s[i]
            nonzero_cols = np.nonzero(np.any(curr_mat != 0, axis=0))[0]
            print("****************************************")
            print("SIMULATION MATRICES")
            if i == (H_s.shape[0] - 1):
                print("Sim Basis Matrix {}".format(i))
            else:
                print("Sim Basis Matrix {}".format(i))
            print("Sim Channels Contained {}".format(nonzero_cols))
            #print("Diag: ", set(list(np.diag(curr_mat))))
            #print("OffDiag: ", set(list(curr_mat[np.triu_indices(dim, k=1)])))
            print("****************************************")
            print()

    return H_s, precision, prec_coeffs

def collect_precision_matrix(H_s, prec_coeffs, P):
    psi_hat = np.sum(np.expand_dims(prec_coeffs, 1)*H_s, 0)
    psi_hat = symmetrize_from_vector(psi_hat, P)
    #precision = (prec_coeffs.reshape(-1, 1, 1)*H_s).sum(0)
    assert is_pos_def(psi_hat), "Not Pos Def"
    return psi_hat

def slow_permutations(upper_indices, nonzero_cols):
    new_left = []
    new_right = []
    for i in range(len(upper_indices[0])):
        left_idx = upper_indices[0][i]
        right_idx = upper_indices[1][i]
        if np.isin(left_idx, nonzero_cols) and np.isin(right_idx, nonzero_cols):
            new_left.append(left_idx)
            new_right.append(right_idx)
        
    return (np.array(new_left), np.array(new_right))

def create_residual_structured(H_s, omega, dim, N, num_indices=4, double_H=False, small_scale=False):
    U = np.zeros((dim, dim))
    # randomly select an H matrix
    rand_H_num = np.random.choice(np.arange(H_s.shape[0]))
    curr_H = symmetrize_from_vector(H_s[rand_H_num], dim=dim)
    if double_H:
        rand_H_num = np.random.choice(np.arange(H_s.shape[0]), size=2, replace=False)
        nonzero_cols = []
        nonzero_rows = []
        for cnum in rand_H_num:
            curr_H = symmetrize_from_vector(H_s[cnum], dim=dim)
            nonzero_cols_curr = np.nonzero(np.any(curr_H != 0, axis=0))[0]
            nonzero_rows_curr = np.nonzero(np.any(curr_H != 0, axis=1))[0]
            nonzero_cols.extend(nonzero_cols_curr)
            nonzero_rows.extend(nonzero_rows_curr)
        nonzero_cols = np.array(nonzero_cols)
        nonzero_rows = np.array(nonzero_rows)
    else:
        nonzero_cols = np.nonzero(np.any(curr_H != 0, axis=0))[0]
        nonzero_rows = np.nonzero(np.any(curr_H != 0, axis=1))[0]
    upper_indices = np.triu_indices(dim, k=1)
    upper_indices = slow_permutations(upper_indices, nonzero_cols)
    if num_indices > len(upper_indices[0]):
        num_indices = len(upper_indices[0])
    #print(len(upper_indices[0]), num_indices)
    # for i in range(len(upper_indices[0])):
    #     print(upper_indices[0][i], upper_indices[1][i])
    rand_four_indices = np.random.choice(np.arange(len(upper_indices[0])), size=num_indices, replace=False)
    w = np.max(np.diag(omega))
    log_dim_over_N = np.log(dim)/N
    root_log_dim = np.sqrt(log_dim_over_N)
    rand_val_zero = -2*w*root_log_dim
    rand_val_one = -w*root_log_dim
    rand_val_two = w*root_log_dim
    rand_val_three = 2*w*root_log_dim
    for idx in rand_four_indices:
        #print(idx)
        i = upper_indices[0][idx]
        j = upper_indices[1][idx]
        pos_neg = np.random.choice(np.arange(num_indices))
        if pos_neg in [0, 1]:
            if not small_scale:
                rand_val = np.random.uniform(rand_val_zero, rand_val_one)
            else:
                rand_val = np.random.uniform(-0.8, -0.6)
        else:
            if not small_scale:
                rand_val = np.random.uniform(rand_val_two, rand_val_three)
            else:
                rand_val = np.random.uniform(0.6, 0.8)
        U[i, j] = rand_val
        U[j, i] = rand_val
    return U

def generate_residual_matrix(H_s, precision, dim, N, num_indices=4, resid_type='unstructured'):
    if resid_type == 'unstructured':
        """
        Randomly sampled indices, no consideration of clusters
        """
        R = create_U_cai(precision, dim=dim, N=N, num_indices=num_indices)
    elif resid_type == 'block':
        """
        Randomly sampled indices, within blocks
        Constrain to i,j that are non-zero
        """
        R = create_residual_structured(H_s=H_s, omega=precision, dim=dim, N=N, num_indices=num_indices)
    return R


def anderson_sim_with_residual(M=2, dim=4, N=500, num_indices=4, resid_type='unstructured', save_path=None):
    print("Simulating Anderson Decomp With Residual Matrix: {}".format(resid_type))
    assert dim % M == 0, "Need dim divisible by M for sake of sampling at the moment"
    H_s, precision_one, prec_coeffs_one = generate_matrices_orthogonal(M=M, dim=dim)
    R = generate_residual_matrix(H_s, 
                                 precision_one, 
                                 dim=dim, 
                                 N=N, 
                                 num_indices=num_indices, 
                                 resid_type=resid_type
                                 )
    #delta = np.abs(np.min(np.linalg.eig(precision_one)[0])) + 0.05
    #delta_times_I = np.eye(dim)*delta
    #precision_one = precision_one + delta_times_I
    if not is_pos_def(precision_one + R):
        eig_vals = np.linalg.eig(precision_one + R)[0]
        correction_vector = np.eye(dim)*np.abs(eig_vals.min()) + 0.05 # pos-def correction
        precision_one = precision_one + correction_vector
    precision_two = precision_one + R
    assert is_pos_def(precision_one), print(np.linalg.eig(precision_one)[0])
    assert is_pos_def(precision_two), print(np.linalg.eig(precision_two)[0])
    
    data_one, C_one = sim_data(covar=inv(precision_one), dim=dim, N=N)
    data_two, C_two = sim_data(covar=inv(precision_two), dim=dim, N=N)
    data_total = np.concatenate((data_one, data_two), axis=1)

    neq_indices = np.where(precision_one != precision_two)
    neq_indices_arr = np.concatenate((np.expand_dims(neq_indices[0],1), np.expand_dims(neq_indices[1], 1)), 1)
    if save_path is not None:
        np.savetxt(os.path.join(save_path, 'changed_indx.csv'), neq_indices_arr.astype(int))
    print("Finished Simulation")
    return data_total.T

def sim_changepoint_mv_normal_orthogonal(sim_scale=0.8, M=2, dim=4, N=500, save_path=None):
    print("Simulating Anderson Decomp Data")
    #assert dim % M == 0, "Need dim divisible by M for sake of sampling at the moment"
    H_s, precision_one, prec_coeffs_one = generate_matrices_orthogonal(M=M, dim=dim)
    prec_coeffs_one = np.ones(M)
    precision_one = collect_precision_matrix(H_s, prec_coeffs_one, dim)
    data_one, C_one = sim_data(covar=inv(precision_one), dim=dim, N=N)
    
    prec_coeffs_two = prec_coeffs_one.copy()
    rand_idx = np.random.choice(np.arange(M))
    prec_coeffs_two[rand_idx] += sim_scale
    precision_two = collect_precision_matrix(H_s, prec_coeffs_two, dim)
    data_two, C_two = sim_data(covar=inv(precision_two), dim=dim, N=N)
    
    data_total = np.concatenate((data_one, data_two), axis=1)
    C_total = np.cov(data_total)
    print("Finished Simulation")
    neq_indices = np.where(precision_one != precision_two)
    neq_indices_arr = np.concatenate((np.expand_dims(neq_indices[0],1), np.expand_dims(neq_indices[1], 1)), 1)
    if save_path is not None:
        np.savetxt(os.path.join(save_path, 'changed_indx.csv'), neq_indices_arr)
    return H_s, data_total

def sim_changepoint_mv_normal_orthogonal_no_change(sim_scale=0.8, M=2, dim=4, N=500, save_path=None):
    print("Simulating Anderson Decomp Data")
    #assert dim % M == 0, "Need dim divisible by M for sake of sampling at the moment"
    H_s, precision_one, prec_coeffs_one = generate_matrices_orthogonal(M=M, dim=dim)
    data_one, C_one = sim_data(covar=inv(precision_one), dim=dim, N=N)
    
    prec_coeffs_two = prec_coeffs_one.copy()
    # prec_coeffs_two[0] += sim_scale#*np.random.choice([-1, 1])
    precision_two = collect_precision_matrix(H_s, prec_coeffs_two, dim)
    data_two, C_two = sim_data(covar=inv(precision_two), dim=dim, N=N)
    
    data_total = np.concatenate((data_one, data_two), axis=1)
    C_total = np.cov(data_total)
    print("Finished Simulation")
    neq_indices = np.where(precision_one != precision_two)
    neq_indices_arr = np.concatenate((np.expand_dims(neq_indices[0],1), np.expand_dims(neq_indices[1], 1)), 1)
    if save_path is not None:
        np.savetxt(os.path.join(save_path, 'changed_indx.csv'), neq_indices_arr)
    return H_s, data_total

def sim_changepoint_mv_normal_orthogonal_mult_coeff(sim_scale=0.8, M=2, dim=4, N=500, num_coeffs_change=1, save_path=None):
    #assert dim % M == 0, "Need dim divisible by M for sake of sampling at the moment"
    H_s, precision_one, prec_coeffs_one = generate_matrices_orthogonal(M=M, dim=dim)
    prec_coeffs_one = np.ones(M)*2.0
    precision_one = collect_precision_matrix(H_s, prec_coeffs_one, dim)
    data_one, C_one = sim_data(covar=inv(precision_one), dim=dim, N=N)
    
    assert num_coeffs_change <= M, "Cannot change more coefficients than exist"
    
    # multiple coeffs to change - sample them randomly
    to_change_coeffs = np.random.choice(np.arange(M), num_coeffs_change, replace=False)
    prec_coeffs_two = prec_coeffs_one.copy()
    for i in range(num_coeffs_change):
        prec_coeffs_two[to_change_coeffs[i]] += np.random.uniform(0.8, 1.8, 1)[0]*np.random.choice([-1, 1])
        
    precision_two = collect_precision_matrix(H_s, prec_coeffs_two, dim)
    data_two, C_two = sim_data(covar=inv(precision_two), dim=dim, N=N)
    data_total = np.concatenate((data_one, data_two), axis=1)
    C_total = np.cov(data_total)
    print("Precision Coefficients Pre-Changepoint: ", prec_coeffs_one)
    print("Precision Coefficients Post-Changepoint: ", prec_coeffs_two)
    neq_indices = np.where(prec_coeffs_one != prec_coeffs_two)
    #neq_indices_arr = np.concatenate((np.expand_dims(neq_indices[0],1), np.expand_dims(neq_indices[1], 1)), 1)
    print(neq_indices)
    if save_path is not None:
        np.savetxt(os.path.join(save_path, 'changed_indx.csv'), neq_indices)
    return H_s, data_total


def get_near_psd(A):
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval <= 0.0] = 1e-8

    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)

def sim_data(covar, dim, N=1000):
    if not is_pos_def(covar):
        print("ALERT: Adjusting covar...")
        covar = get_near_psd(covar)
    
    assert is_symmetric(covar), is_pos_def(covar)
    data_sim = np.random.multivariate_normal(np.zeros(dim), covar, N).T
    #data_sim = (data_sim - data_sim.mean())/data_sim.std()
    C = np.cov(data_sim)
    
    return data_sim, C

def sim_changepoint_mv_normal_no_decomp(dim, N, num_coeffs_change=1, scale=0.8, save_path=None):
    print("Simulating Data with No Decomp")
    C_one = make_spd_matrix(dim)
    data_one = np.random.multivariate_normal(np.zeros(dim), C_one, N)
    C_two = C_one.copy()
    for _ in range(num_coeffs_change):
        rand_i_j = np.random.choice(np.arange(dim), 2, replace=False)
        i, j = rand_i_j[0], rand_i_j[1]
        C_two = C_two.copy()
        val = C_two[i,j]
        val += scale
        C_two[i, j] = val
        C_two[j, i] = val
    
        # keep it pos_def
        adjustment = np.abs(np.linalg.eig(C_two)[0].min()) + 1e-12
        C_two += np.eye(dim)*adjustment
    
    data_two = np.random.multivariate_normal(np.zeros(dim), C_two, N)
    data_total = np.concatenate((data_one, data_two), axis=0)
    
    assert is_pos_def(C_one)
    assert is_symmetric(C_one)
    assert is_pos_def(C_two)
    assert is_symmetric(C_two)
    assert sum(np.diag(C_one) <= 0) == 0
    assert sum(np.diag(C_two) <= 0) == 0
    print("Finished Simulation")
    neq_indices = np.where(C_one != C_two)
    neq_indices_arr = np.concatenate((np.expand_dims(neq_indices[0],1), np.expand_dims(neq_indices[1], 1)), 1)
    if save_path is not None:
        np.savetxt(os.path.join(save_path, 'changed_indx.csv'), neq_indices_arr)
    return data_total

def get_scale_val(C, dim, scale_factor=50):
    w = np.max(np.diag(C))
    log_dim_over_N = np.log(dim)/scale_factor
    root_log_dim = np.sqrt(log_dim_over_N)
    rand_val_zero = -2*w*root_log_dim
    rand_val_one = -w*root_log_dim
    rand_val_two = w*root_log_dim
    rand_val_three = 2*w*root_log_dim
    pos_neg = np.random.choice(np.arange(4))
    if pos_neg in [0, 1]:
        rand_val = np.random.uniform(rand_val_zero, rand_val_one)
    else:
        rand_val = np.random.uniform(rand_val_two, rand_val_three)
    #print(rand_val_zero, rand_val_one, rand_val_two, rand_val_three)
    return rand_val

def sim_changepoint_mv_normal_cholesky(dim, N, num_coeffs_change=1, scale=0.8, save_path=None, sparse=False):
    print("Simulating Cholesky Decomp Data")
    L_one = make_spd_matrix(dim)
    L_one = np.tril(L_one)
    L_one /= np.abs(L_one.max())
    if sparse:
        print('Sparse version')
        top_indices = np.tril_indices(L_one.shape[0], k=-1)
        randomly_zero_out = np.random.choice(np.arange(len(top_indices[0])), size=int(len(top_indices[0])*0.75), replace=False)
        zeroed_i = top_indices[0][randomly_zero_out]
        zeroed_j = top_indices[1][randomly_zero_out]
        zipped = list(zip(zeroed_i, zeroed_j))
        for i,j in zipped:
            L_one[i,j] = 0.0
    np.fill_diagonal(L_one, 1.0)
    C_one = L_one.dot(L_one.T)
    #C_one = C_one/np.abs(C_one.max())
    data_one = np.random.multivariate_normal(np.zeros(dim), inv(C_one), N)
    L_two = L_one.copy()
    for _ in range(num_coeffs_change):
        rand_i_j = np.random.choice(np.arange(dim), 2, replace=False)
        i, j = rand_i_j[0], rand_i_j[1]
        if i >= j:
            val = L_two[i, j]
        else:
            val = L_two[j, i]
        val += get_scale_val(C_one, dim)
        # val += scale
        # no need for symmetric change - it's a cholesky
        if i >= j:
            L_two[i, j] = val
        else:
            L_two[j, i] = val
    C_two = L_two.dot(L_two.T)
    #adjustment = np.abs(np.linalg.eig(C_two)[0].min()) + 1e-12
    #C_two += np.eye(dim)*adjustment
    #C_two = C_two/np.abs(C_two.max())
    data_two = np.random.multivariate_normal(np.zeros(dim), inv(C_two), N)
    data_total = np.concatenate((data_one, data_two), axis=0)
    
    assert is_pos_def(C_one)
    assert is_symmetric(C_one)
    assert is_pos_def(C_two)
    assert is_symmetric(C_two)
    assert sum(np.diag(inv(C_one)) <= 0) == 0
    assert sum(np.diag(inv(C_two)) <= 0) == 0
    neq_indices = np.where(C_one != C_two)
    neq_indices_arr = np.concatenate((np.expand_dims(neq_indices[0],1), np.expand_dims(neq_indices[1], 1)), 1)
    if save_path is not None:
        np.savetxt(os.path.join(save_path, 'changed_indx.csv'), neq_indices_arr)
    print("Finished Simulation")
    return data_total

def sim_changepoint_mv_normal_cholesky_no_change(dim, N, num_coeffs_change=1, scale=0.8, save_path=None, sparse=False):
    print("Simulating Cholesky Decomp Data")
    L_one = make_spd_matrix(dim)
    L_one = np.tril(L_one)
    L_one /= np.abs(L_one.max())
    if sparse:
        print("Sparse version")
        top_indices = np.tril_indices(L_one.shape[0], k=-1)
        randomly_zero_out = np.random.choice(np.arange(len(top_indices[0])), size=int(len(top_indices[0])*0.75), replace=False)
        zeroed_i = top_indices[0][randomly_zero_out]
        zeroed_j = top_indices[1][randomly_zero_out]
        zipped = list(zip(zeroed_i, zeroed_j))
        for i,j in zipped:
            L_one[i,j] = 0.0
    np.fill_diagonal(L_one, 1.0)
    C_one = L_one.dot(L_one.T)
    #C_one = C_one/np.abs(C_one.max())
    data_one = np.random.multivariate_normal(np.zeros(dim), inv(C_one), N)
    L_two = L_one.copy()
    for _ in range(num_coeffs_change):
        rand_i_j = np.random.choice(np.arange(dim), 2, replace=False)
        i, j = rand_i_j[0], rand_i_j[1]
        L_two = L_two.copy()
        if i >= j:
            val = L_two[i, j]
        else:
            val = L_two[j,i]
        val += 0.0#get_scale_val(C_one, dim)
        # val += scale
        # no need for symmetric change - it's a cholesky
        if i >= j:
            L_two[i, j] = val
        else:
            L_two[j,i] = val
    C_two = L_two.dot(L_two.T)
    #adjustment = np.abs(np.linalg.eig(C_two)[0].min()) + 1e-12
    #C_two += np.eye(dim)*adjustment
    #C_two = C_two/np.abs(C_two.max())
    data_two = np.random.multivariate_normal(np.zeros(dim), inv(C_two), N)
    data_total = np.concatenate((data_one, data_two), axis=0)
    
    assert is_pos_def(C_one)
    assert is_symmetric(C_one)
    assert is_pos_def(C_two)
    assert is_symmetric(C_two)
    assert sum(np.diag(inv(C_one)) <= 0) == 0
    assert sum(np.diag(inv(C_two)) <= 0) == 0
    neq_indices = np.where(C_one != C_two)
    neq_indices_arr = np.concatenate((np.expand_dims(neq_indices[0],1), np.expand_dims(neq_indices[1], 1)), 1)
    if save_path is not None:
        np.savetxt(os.path.join(save_path, 'changed_indx.csv'), neq_indices_arr)
    print("Finished Simulation")
    return data_total

def sim_changepoint_mv_normal_ldlt(dim, N, num_coeffs_change=1, scale=0.8, save_path=None):
    print("Simulating LDLT Decomp Data")
    L_one = make_spd_matrix(dim)
    L_one = np.tril(L_one)
    L_one /= np.abs(L_one.max())
    np.fill_diagonal(L_one, 1.0)
    D = np.diag(np.ones(dim)*2.0)
    C_one = L_one@D@(L_one.T)
    C_one = C_one#/np.abs(C_one.max())
    data_one = np.random.multivariate_normal(np.zeros(dim), inv(C_one), N)
    L_two = L_one.copy()
    for _ in range(num_coeffs_change):
        rand_i_j = np.random.choice(np.arange(dim), 2, replace=False)
        i, j = rand_i_j[0], rand_i_j[1]
        assert i != j, "Only the off-diagonal should be changed"
        L_two = L_two.copy()
        if i >= j:
            val = L_two[i, j]
        else:
            val = L_two[j,i]
        val += get_scale_val(C_one, dim)
        # val += scale
        # no need for symmetric change - it's a cholesky
        if i >= j:
            L_two[i, j] = val
        else:
            L_two[j, i] = val
    
    np.fill_diagonal(L_two, 1.0)
    C_two = L_two@D@(L_two.T)
    C_two = C_two#/np.abs(C_two.max())
    data_two = np.random.multivariate_normal(np.zeros(dim), inv(C_two), N)
    data_total = np.concatenate((data_one, data_two), axis=0)

    #print(np.abs(C_one-C_two).max())
    #print(C_one.max(), C_two.max())
    assert is_pos_def(C_one)
    assert is_symmetric(C_one)
    assert is_pos_def(C_two)
    assert is_symmetric(C_two)
    assert sum(np.diag(inv(C_one)) <= 0) == 0
    assert sum(np.diag(inv(C_two)) <= 0) == 0
    # print(C_one.max())
    # print(C_two.max())
    # print(inv(C_one).max())
    # print(inv(C_two).max())
    print("Finished Simulation")
    neq_indices = np.where(C_one != C_two)
    neq_indices_arr = np.concatenate((np.expand_dims(neq_indices[0],1), np.expand_dims(neq_indices[1], 1)), 1)
    if save_path is not None:
        np.savetxt(os.path.join(save_path, 'changed_indx.csv'), neq_indices_arr)
    return data_total

def sim_changepoint_var_process(dim, N, num_coeffs_change, scale=0.5, save_path=None):
    # simulate this where we change the noise covariance via a LDLT decomposition on precision
    # VAR(1)
    print("Simulating VAR Data")
    coeffs = np.random.uniform(low=-0.1, high=0.1, size=(1, dim, dim))
    L_one = make_spd_matrix(dim)
    L_one = np.tril(L_one)
    np.fill_diagonal(L_one, 1.0)
    D = np.diag(np.ones(dim)*2.0)
    C_one = L_one@D@(L_one.T)
    L_two = L_one.copy()
    for _ in range(num_coeffs_change):
        rand_i_j = np.random.choice(np.arange(dim), 2, replace=False)
        i, j = rand_i_j[0], rand_i_j[1]
        assert i != j, "Only the off-diagonal should be changed"
        L_two = L_two.copy()
        val = L_two[i, j]
        val += scale
        # no need for symmetric change - it's a cholesky
        L_two[i, j] = val

    np.fill_diagonal(L_two, 1.0)
    C_two = L_two@D@(L_two.T)

    assert is_pos_def(C_one)
    assert is_symmetric(C_one)
    assert is_pos_def(C_two)
    assert is_symmetric(C_two)

    var_one = VARProcess(coefs=coeffs, coefs_exog=np.zeros(1), sigma_u=inv(C_one))
    var_two = VARProcess(coefs=coeffs, coefs_exog=np.zeros(1), sigma_u=inv(C_two))

    data_one = var_one.simulate_var(steps=N)
    data_two = var_two.simulate_var(steps=N)

    data_total = np.concatenate((data_one, data_two), axis=0)
    print("Finished Simulation")
    neq_indices = np.where(C_one != C_two)
    neq_indices_arr = np.concatenate((np.expand_dims(neq_indices[0],1), np.expand_dims(neq_indices[1], 1)), 1)
    if save_path is not None:
        np.savetxt(os.path.join(save_path, 'changed_indx.csv'), neq_indices_arr)
    return data_total

def create_matrix_kesh(p, d):
    assert d < p, "Too many zero entries"
    U = np.random.normal(size=(p, p))
    for i in range(p):
        arr = np.array([1]*d + [0]*(p-d))
        np.random.shuffle(arr)
        U[i, :] = U[i, :]*arr
    
    first = 1/np.linalg.norm(np.dot(U, U.T), ord=np.inf)
    second = np.dot(U, U.T)
    H = np.dot(first, second)

    return H

def changepoint_kesh_model(p, d, N, beta=0.2, lambda_0=0.1):
    H = create_matrix_kesh(p=p, d=d)
    omega_bc = H + lambda_0*np.eye(p)
    omega_ac = (1+beta)*omega_bc

    data_one = np.random.multivariate_normal(np.zeros(p), inv(omega_bc), N)
    data_two = np.random.multivariate_normal(np.zeros(p), inv(omega_bc), N)

    data_total = np.concatenate((data_one, data_two), axis=0)

    return data_total, omega_bc

def sim_changepoint_cai_model_one(dim):
    D_diag = np.random.uniform(0.5, 2.5, dim)
    D = np.diag(D_diag)
    omega = np.zeros((dim, dim))
    for i in range(dim):
        omega[i, i] = 1
        if i < dim-1:
            omega[i, i+1] = 0.6
            omega[i+1, i] = 0.6
            if i < dim-2:
                omega[i, i+2] = 0.3
                omega[i+2, i] = 0.3
    
    omega = np.sqrt(D)@omega@np.sqrt(D)

    return omega

def sim_changepoint_cai_model_three(dim):
    D_diag = np.random.uniform(0.5, 2.5, dim)
    D = np.diag(D_diag)
    omega = np.zeros((dim, dim))
    for i in range(dim-1):
        for j in range(i, dim):
            b_val = 0.8*np.random.binomial(1, 0.05)
            omega[i, j] = b_val
            omega[j, i] = b_val
    
    np.fill_diagonal(omega, 1.0)
    delta = np.abs(np.min(np.linalg.eig(omega)[0])) + 0.05
    delta_times_I = np.eye(dim)*delta
    omega_plus = omega + delta_times_I
    middle = omega_plus/(1+delta)
    omega = np.power(D, 0.5)@middle@np.power(D, 0.5)
    assert(is_pos_def(omega))
    
    return omega

def sim_changepoint_cai_model_four(dim):
    D_diag = np.random.uniform(0.5, 2.5, dim)
    D = np.diag(D_diag)
    sigma = np.zeros((dim, dim))
    for k in range(dim//2):
        for i in range(2*(k-1), 2*k):
            for j in range(2*(k-1), 2*k):
                #print(k, i, j)
                if i != j:
                    sigma[i,j] = 0.5
                    sigma[j,i] = 0.5
    np.fill_diagonal(sigma, 1.0)
    delta = np.abs(np.min(np.linalg.eig(sigma)[0])) + 0.05
    delta_times_I = np.eye(dim)*delta
    top = sigma + delta_times_I
    bottom = 1 + delta
    omega = np.sqrt(D)@inv(top/bottom)@np.sqrt(D)
    assert is_pos_def(omega)
    
    return omega
    

def create_U_cai(omega, dim, N, num_indices=4):
    U = np.zeros((dim, dim))
    upper_indices = np.triu_indices(dim, k=1)
    rand_four_indices = np.random.choice(np.arange(len(upper_indices[0])), size=num_indices)
    w = np.max(np.diag(omega))
    log_dim_over_N = np.log(dim)/N
    root_log_dim = np.sqrt(log_dim_over_N)
    rand_val_zero = -2*w*root_log_dim
    rand_val_one = -w*root_log_dim
    rand_val_two = w*root_log_dim
    rand_val_three = 2*w*root_log_dim
    for idx in rand_four_indices:
        i = upper_indices[0][idx]
        j = upper_indices[1][idx]
        pos_neg = np.random.choice(np.arange(num_indices))
        if pos_neg in [0, 1]:
            rand_val = np.random.uniform(rand_val_zero, rand_val_one)
        else:
            rand_val = np.random.uniform(rand_val_two, rand_val_three)
        U[i, j] = rand_val
        U[j, i] = rand_val
    return U

def create_U_cai_no_change(omega, dim, N):
    U = np.zeros((dim, dim))
    upper_indices = np.triu_indices(dim, k=1)
    rand_four_indices = np.random.choice(np.arange(len(upper_indices[0])), size=4)
    w = np.max(np.diag(omega))
    log_dim_over_N = np.log(dim)/N
    root_log_dim = np.sqrt(log_dim_over_N)
    rand_val_zero = -2*w*root_log_dim
    rand_val_one = -w*root_log_dim
    rand_val_two = w*root_log_dim
    rand_val_three = 2*w*root_log_dim
    for idx in rand_four_indices:
        i = upper_indices[0][idx]
        j = upper_indices[1][idx]
        pos_neg = np.random.choice(np.arange(4))
        if pos_neg in [0, 1]:
            rand_val = 0.0#np.random.uniform(rand_val_zero, rand_val_one)
        else:
            rand_val = 0.0#np.random.uniform(rand_val_two, rand_val_three)
        U[i, j] = rand_val
        U[j, i] = rand_val
    return U

def simulate_changepoint_cai(omega, U, N=1000):
    # simulate 1000 points of data, but need to chop it to 100, 100 for the Cai comparison
    dim = omega.shape[0]
    min_eig_val_omega = np.linalg.eig(omega)[0].min()
    min_eig_val_omega_plus = np.linalg.eig(omega+U)[0].min()
    delta = np.abs(np.minimum(min_eig_val_omega, min_eig_val_omega_plus)) + 0.05
    mat_shift = np.eye(dim)*delta
    precision_one = omega + mat_shift
    precision_two = omega + U + mat_shift
    assert(is_pos_def(precision_one))
    assert(is_pos_def(precision_two))
    data_one = np.random.multivariate_normal(np.zeros(dim), inv(precision_one), N)
    data_two = np.random.multivariate_normal(np.zeros(dim), inv(precision_two), N)
    data_full = np.concatenate((data_one, data_two), axis=0)
    return data_full, precision_one, precision_two

def changepoint_cai_model_one(args, dim, N=100, save_path=None):
    print("Simulating Cai Model One")
    omega = sim_changepoint_cai_model_one(dim=dim)
    U = create_U_cai(omega, dim=dim, N=N, num_indices=args.num_indices)
    data_full, precision_one, precision_two = simulate_changepoint_cai(omega, U, N=N)
    neq_indices = np.where(precision_one != precision_two)
    neq_indices_arr = np.concatenate((np.expand_dims(neq_indices[0],1), np.expand_dims(neq_indices[1], 1)), 1)
    if save_path is not None:
        np.savetxt(os.path.join(save_path, 'changed_indx.csv'), neq_indices_arr)
    print("Finished Simulation")
    return data_full

def changepoint_cai_model_one_no_change(dim, N=100, save_path=None):
    print("Simulating Cai Model One")
    omega = sim_changepoint_cai_model_one(dim=dim)
    U = create_U_cai_no_change(omega, dim=dim, N=N)
    data_full, precision_one, precision_two = simulate_changepoint_cai(omega, U, N=N)
    neq_indices = np.where(precision_one != precision_two)
    neq_indices_arr = np.concatenate((np.expand_dims(neq_indices[0],1), np.expand_dims(neq_indices[1], 1)), 1)
    if save_path is not None:
        np.savetxt(os.path.join(save_path, 'changed_indx.csv'), neq_indices_arr)
    print("Finished Simulation")
    return data_full, precision_one


def changepoint_cai_model_three(args, dim, N=100, save_path=None):
    print("Simulating Cai Model Three")
    omega = sim_changepoint_cai_model_three(dim=dim)
    U = create_U_cai(omega, dim=dim, N=N, num_indices=args.num_indices)
    data_full, precision_one, precision_two = simulate_changepoint_cai(omega, U, N=N)
    neq_indices = np.where(precision_one != precision_two)
    neq_indices_arr = np.concatenate((np.expand_dims(neq_indices[0],1), np.expand_dims(neq_indices[1], 1)), 1)
    if save_path is not None:
        np.savetxt(os.path.join(save_path, 'changed_indx.csv'), neq_indices_arr)
    print("Finished Simulation")
    return data_full

def changepoint_cai_model_three_no_change(dim, N=100, save_path=None):
    print("Simulating Cai Model Three")
    omega = sim_changepoint_cai_model_three(dim=dim)
    U = create_U_cai_no_change(omega, dim=dim, N=N)
    data_full, precision_one, precision_two = simulate_changepoint_cai(omega, U, N=N)
    neq_indices = np.where(precision_one != precision_two)
    neq_indices_arr = np.concatenate((np.expand_dims(neq_indices[0],1), np.expand_dims(neq_indices[1], 1)), 1)
    if save_path is not None:
        np.savetxt(os.path.join(save_path, 'changed_indx.csv'), neq_indices_arr)
    print("Finished Simulation")
    return data_full

def changepoint_cai_model_four(args, dim, N=100, save_path=None):
    print("Simulating Cai Model Four")
    omega = sim_changepoint_cai_model_four(dim=dim)
    U = create_U_cai(omega, dim=dim, N=N, num_indices=args.num_indices)
    data_full, precision_one, precision_two = simulate_changepoint_cai(omega, U, N=N)
    neq_indices = np.where(precision_one != precision_two)
    neq_indices_arr = np.concatenate((np.expand_dims(neq_indices[0],1), np.expand_dims(neq_indices[1], 1)), 1)
    if save_path is not None:
        np.savetxt(os.path.join(save_path, 'changed_indx.csv'), neq_indices_arr)
    print("Finished Simulation")
    return data_full

def changepoint_cai_model_four_no_change(dim, N=100, save_path=None):
    print("Simulating Cai Model Four")
    omega = sim_changepoint_cai_model_four(dim=dim)
    U = create_U_cai_no_change(omega, dim=dim, N=N)
    data_full, precision_one, precision_two = simulate_changepoint_cai(omega, U, N=N)
    neq_indices = np.where(precision_one != precision_two)
    neq_indices_arr = np.concatenate((np.expand_dims(neq_indices[0],1), np.expand_dims(neq_indices[1], 1)), 1)
    if save_path is not None:
        np.savetxt(os.path.join(save_path, 'changed_indx.csv'), neq_indices_arr)
    print("Finished Simulation")
    return data_full



