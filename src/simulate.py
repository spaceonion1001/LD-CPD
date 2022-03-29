import numpy as np
from sklearn.datasets import make_spd_matrix
from sklearn.preprocessing import scale
from utils import is_pos_def, is_symmetric
from numpy.linalg import inv as inv

def generate_matrices_orthogonal(prec_coeffs=None, M=2, dim=4):
    H_s = []
    if not prec_coeffs:
        prec_coeffs = np.random.rand(M)
    precision = np.zeros((dim, dim))
    H_s_stacked = np.zeros((dim**2, M))
    mat = make_spd_matrix(dim)
    indxes = np.random.choice(np.arange(dim), (M, dim//M), replace=False)
    for i in range(M):
        A = np.zeros((dim, dim))
        for idx in indxes[i]:
            for idx2 in indxes[i]:
                A[idx, idx2] = mat[idx, idx2]
        H_s.append(A)
        precision += prec_coeffs[i]*A
        H_s_stacked[:, i] = A.flatten()
    H_s = np.array(H_s)
    
    # ensure basis matrices are linearly independent
    assert np.linalg.matrix_rank(H_s_stacked) == M, "Not Linearly Independent basis matrices"
    
    # ensure it's actually symmetric
    # minimal modification on scale of 1e-15
    precision_corrected = (precision + precision.T)/2
    
    return H_s, precision_corrected, prec_coeffs

def collect_precision_matrix(H_s, prec_coeffs):
    precision = (prec_coeffs.reshape(-1, 1, 1)*H_s).sum(0)
    
    return precision

def sim_changepoint_mv_normal_orthogonal(M=2, dim=4, N=500):
    assert dim % M == 0, "Need dim divisible by M for sake of sampling at the moment"
    H_s, precision_one, prec_coeffs_one = generate_matrices_orthogonal(M=M, dim=dim)
    data_one, C_one = sim_data(covar=inv(precision_one), dim=dim, N=N)
    
    prec_coeffs_two = prec_coeffs_one.copy()
    prec_coeffs_two[0] += 0.8
    precision_two = collect_precision_matrix(H_s, prec_coeffs_two)
    data_two, C_two = sim_data(covar=inv(precision_two), dim=dim, N=N)
    
    data_total = np.concatenate((data_one, data_two), axis=1)
    C_total = np.cov(data_total)
    
    return H_s, data_total

def sim_changepoint_mv_normal_orthogonal_mult_coeff(M=2, dim=4, N=500, num_coeffs_change=1):
    assert dim % M == 0, "Need dim divisible by M for sake of sampling at the moment"
    H_s, precision_one, prec_coeffs_one = generate_matrices_orthogonal(M=M, dim=dim)
    data_one, C_one = sim_data(covar=inv(precision_one), dim=dim, N=N)
    
    assert num_coeffs_change <= M, "Cannot change more coefficients than exist"
    
    # multiple coeffs to change - sample them randomly
    to_change_coeffs = np.random.choice(np.arange(M), num_coeffs_change, replace=False)
    prec_coeffs_two = prec_coeffs_one.copy()
    for i in range(num_coeffs_change):
        prec_coeffs_two[to_change_coeffs[i]] += np.random.uniform(0.1, 0.3, 1)[0]
        
    precision_two = collect_precision_matrix(H_s, prec_coeffs_two)
    data_two, C_two = sim_data(covar=inv(precision_two), dim=dim, N=N)
    
    data_total = np.concatenate((data_one, data_two), axis=1)
    C_total = np.cov(data_total)
    print("Precision Coefficients Pre-Changepoint: ", prec_coeffs_one)
    print("Precision Coefficients Post-Changepoint: ", prec_coeffs_two)
    
    return H_s, data_total

def sim_data(covar, dim, N=1000):
    assert is_symmetric(covar), is_pos_def(covar)
    data_sim = np.random.multivariate_normal(np.zeros(dim), covar, N).T
    data_sim = data_sim - data_sim.mean()
    C = np.cov(data_sim)
    
    return data_sim, C

def sim_changepoint_mv_normal_no_decomp(dim, N, num_coeffs_change=1, scale=0.8):
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
    
    return data_total.T

def sim_changepoint_mv_normal_cholesky(dim, N, num_coeffs_change=1):
    L_one = make_spd_matrix(dim)
    C_one = L_one.dot(L_one.T)
    data_one = np.random.multivariate_normal(np.zeros(dim), C_one, N)
    L_two = L_one.copy()
    for _ in range(num_coeffs_change):
        rand_i_j = np.random.choice(np.arange(dim), 2, replace=False)
        i, j = rand_i_j[0], rand_i_j[1]
        L_two = L_two.copy()
        val = L_two[i, j]
        val += scale
        # no need for symmetric change - it's a cholesky
        L_two[i, j] = val

    C_two = L_two.dot(L_two.T)

    data_two = np.random.multivariate_normal(np.zeros(dim), C_two, N)
    data_total = np.concatenate((data_one, data_two), axis=0)
    
    assert is_pos_def(C_one)
    assert is_symmetric(C_one)
    assert is_pos_def(C_two)
    assert is_symmetric(C_two)
    
    return data_total.T

