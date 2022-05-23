import numpy as np
from sklearn.datasets import make_spd_matrix
from sklearn.preprocessing import scale
from utils import is_pos_def, is_symmetric
from numpy.linalg import inv as inv
from statsmodels.tsa.vector_ar.var_model import VARProcess
import matplotlib.pyplot as plt
import os

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

def sim_changepoint_mv_normal_orthogonal(sim_scale=0.8, M=2, dim=4, N=500, save_path=None):
    print("Simulating Anderson Decomp Data")
    assert dim % M == 0, "Need dim divisible by M for sake of sampling at the moment"
    H_s, precision_one, prec_coeffs_one = generate_matrices_orthogonal(M=M, dim=dim)
    data_one, C_one = sim_data(covar=inv(precision_one), dim=dim, N=N)
    
    prec_coeffs_two = prec_coeffs_one.copy()
    prec_coeffs_two[0] += sim_scale#*np.random.choice([-1, 1])
    precision_two = collect_precision_matrix(H_s, prec_coeffs_two)
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
    assert dim % M == 0, "Need dim divisible by M for sake of sampling at the moment"
    H_s, precision_one, prec_coeffs_one = generate_matrices_orthogonal(M=M, dim=dim)
    data_one, C_one = sim_data(covar=inv(precision_one), dim=dim, N=N)
    
    prec_coeffs_two = prec_coeffs_one.copy()
    # prec_coeffs_two[0] += sim_scale#*np.random.choice([-1, 1])
    precision_two = collect_precision_matrix(H_s, prec_coeffs_two)
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
    assert dim % M == 0, "Need dim divisible by M for sake of sampling at the moment"
    H_s, precision_one, prec_coeffs_one = generate_matrices_orthogonal(M=M, dim=dim)
    data_one, C_one = sim_data(covar=inv(precision_one), dim=dim, N=N)
    
    assert num_coeffs_change <= M, "Cannot change more coefficients than exist"
    
    # multiple coeffs to change - sample them randomly
    to_change_coeffs = np.random.choice(np.arange(M), num_coeffs_change, replace=False)
    prec_coeffs_two = prec_coeffs_one.copy()
    for i in range(num_coeffs_change):
        prec_coeffs_two[to_change_coeffs[i]] += np.random.uniform(0.8, 1.5, 1)[0]*np.random.choice([-1, 1])
        
    precision_two = collect_precision_matrix(H_s, prec_coeffs_two)
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

def sim_data(covar, dim, N=1000):
    assert is_symmetric(covar), is_pos_def(covar)
    data_sim = np.random.multivariate_normal(np.zeros(dim), covar, N).T
    data_sim = data_sim - data_sim.mean()
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

def get_scale_val(C, dim):
    w = np.max(np.diag(C))
    log_dim_over_N = np.log(dim)/50
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
    

def create_U_cai(omega, dim, N):
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

def changepoint_cai_model_one(dim, N=100, save_path=None):
    print("Simulating Cai Model One")
    omega = sim_changepoint_cai_model_one(dim=dim)
    U = create_U_cai(omega, dim=dim, N=N)
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
    return data_full


def changepoint_cai_model_three(dim, N=100, save_path=None):
    print("Simulating Cai Model Three")
    omega = sim_changepoint_cai_model_three(dim=dim)
    U = create_U_cai(omega, dim=dim, N=N)
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

def changepoint_cai_model_four(dim, N=100, save_path=None):
    print("Simulating Cai Model Four")
    omega = sim_changepoint_cai_model_four(dim=dim)
    U = create_U_cai(omega, dim=dim, N=N)
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



