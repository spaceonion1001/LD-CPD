import numpy as np
from numpy.linalg import inv
import cvxpy as cp
from utils import lasso_likelihood, vectorize_matrix, symmetrize_from_vector, symmetrize_from_vector_alt, is_pos_def
from numba import jit
from scipy.optimize import line_search
from sklearn.utils.extmath import fast_logdet
import pdb
import copy

class JacWrapper(object):
    def __init__(self, H_s, C, dim):
        self.H_s = H_s
        self.C = C
        self.dim = dim
    
    def boyd_likelihood(self, x):
        alphas = x
        new_psi_hat = calc_psi_hat(alphas=alphas, H_s=self.H_s, dim=self.dim)
        P = self.dim
        _, first_term = np.linalg.slogdet(new_psi_hat)
        likelihood = first_term - np.trace(new_psi_hat@self.C) - P*np.log(2*np.pi)

        return -likelihood
    
    def calc_jacobian(self, x):
        alphas = x
        M = self.H_s.shape[0]
        psi_hat = calc_psi_hat(alphas, self.H_s, self.dim)
        inv_psi_hat = inv(psi_hat)
        jacobian = np.zeros((M, ))
        for j in range(M):
            H_j = symmetrize_from_vector(self.H_s[j], self.dim)
            first_term = np.trace(inv_psi_hat.dot(H_j))
            second_term = np.trace(H_j.dot(self.C))
            jacobian[j] = -1*(first_term - second_term) # TODO is this sign right?!
        
        return jacobian


#@jit(nopython=True)
def boyd_likelihood(x, H_s, C, dim):
    """
    This is negative log likelihood for minimization purposes
    """
    alphas = x
    new_psi_hat = calc_psi_hat(alphas=alphas, H_s=H_s, dim=dim)
    assert is_pos_def(new_psi_hat), pdb.set_trace()
    P = dim
    #_, first_term = np.linalg.slogdet(new_psi_hat)
    first_term = fast_logdet(new_psi_hat)
    likelihood = first_term - np.trace(new_psi_hat@C) - P*np.log(2*np.pi)

    return -likelihood

#@jit(nopython=True)
def boyd_likelihood_dc(x, H, C, dim):
    psi_hat = x*H
    P = dim

    first_term = fast_logdet(psi_hat)
    #first_term = np.linalg.slogdet(psi_hat)[1]
    likelihood = first_term - np.trace(psi_hat@C) - P*np.log(2*np.pi)

    return -likelihood
    

#@jit(nopython=True)
def calc_hessian(alphas, H_s, dim):
    M = H_s.shape[0]
    psi_hat = calc_psi_hat(alphas, H_s, dim)
    inv_psi_hat = inv(psi_hat)
    hessian = np.zeros((M, M))
    for j in range(M):
        for k in range(M):
            H_j = symmetrize_from_vector(H_s[j], dim)
            H_k = symmetrize_from_vector(H_s[k], dim)
            prod = inv_psi_hat.dot(H_j).dot(inv_psi_hat).dot(H_k)
            hessian[j,k] = np.trace(prod) # double check sign
    assert is_pos_def(hessian)
    return hessian

#@jit(nopython=True)
def calc_jacobian(x, *args, H_s, C, dim):
    alphas = x
    M = H_s.shape[0]
    psi_hat = calc_psi_hat(alphas, H_s, dim)
    inv_psi_hat = inv(psi_hat)
    jacobian = np.zeros((M, ))
    for j in range(M):
        H_j = symmetrize_from_vector(H_s[j], dim)
        first_term = np.trace(inv_psi_hat.dot(H_j))
        second_term = np.trace(H_j.dot(C))
        jacobian[j] = -1*(first_term - second_term) # double check sign
    
    return jacobian


@jit(nopython=True, cache=True)
def np_all_axis0(x):
    """Numba compatible version of np.all(x, axis=0)."""
    out = np.ones(x.shape[1], dtype=np.bool8)
    for i in range(x.shape[0]):
        out = np.logical_and(out, x[i, :])
    return out

@jit(nopython=True, cache=True)
def np_all_axis1(x):
    """Numba compatible version of np.all(x, axis=1)."""
    out = np.ones(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_and(out, x[:, i])
    return out

@jit(nopython=True, cache=True)
def calc_jacobian_hessian_dc(x, H, C, dim):
    """
    Divide and conquer derivative of likelihood function
    x = single alpha
    H = single cluster from block matrix
    C = single cluster sample covariance
    dim = dimensionality of problem
    """
    psi_hat = x*H
    psi_hat = symmetrize_from_vector(psi_hat, dim)
    #psi_hat = psi_hat[:, ~np.all(psi_hat == 0, axis=0)]
    psi_hat = psi_hat[:, ~np_all_axis0(psi_hat == 0)]
    #psi_hat = psi_hat[~np.all(psi_hat == 0, axis=1), :]
    psi_hat = psi_hat[~np_all_axis1(psi_hat == 0), :]
    assert psi_hat.shape[0] == C.shape[0]

    inv_psi_hat = inv(psi_hat)
    H_curr = symmetrize_from_vector(H, dim)
    #H_curr = H_curr[:, ~np.all(H_curr == 0, axis=0)]
    #H_curr = H_curr[~np.all(H_curr == 0, axis=1), :]
    H_curr = H_curr[:, ~np_all_axis0(H_curr == 0)]
    H_curr = H_curr[~np_all_axis1(H_curr == 0), :]

    first_term = np.trace(inv_psi_hat.dot(H_curr))
    second_term = np.trace(H_curr.dot(C))

    jacobian = -1*(first_term - second_term) # scalar value
    hess = np.trace(inv_psi_hat.dot(H_curr).dot(inv_psi_hat).dot(H_curr)) # no pairwise entries or other clusters - scalar value

    return jacobian, hess

#@jit(nopython=True)
def perform_line_search_dc(local_slope, newton_direction, alpha, H, C, dim, tau, control_factor):
    t = -control_factor*local_slope
    #print("t {}".format(t))
    max_iters = 500
    # calculate maximum step size heuristic
    new_alph_temp = alpha + 1.0*newton_direction
    lr_val = 1.0
    if not new_alph_temp <= 0:
        lr_val = 1.0
    else:
        while np.array(alpha + lr_val*newton_direction <= 1e-8).any():
            lr_val = lr_val*0.95
        # alph_diff = alphas - new_alph_temp
        # min_val = np.min(new_alph_temp) # most negative value
        # min_idx = np.argmin(new_alph_temp)
        # # TODO FIX THIS LINE
        # """
        # INCOMPLETE
        # """
        # lr_val = -(alphas[min_idx]/newton_direction[min_idx])*0.5
        # """
        # """
    #print("\nLINE SEARCH\n")
    for j in range(max_iters):
        f_x = boyd_likelihood_dc(x=alpha, H=H, C=C, dim=C.shape[0])
        new_alphas = alpha+lr_val*newton_direction
        f_x_update = boyd_likelihood_dc(x=new_alphas, H=H, C=C, dim=C.shape[0])
        fx_diff = f_x - f_x_update
        #print("***********")
        #print("ITER {}".format(j))
        #print("FX Diff {} Orig {} New {}".format(fx_diff, f_x, f_x_update))
        #print("New Alphas {}".format(new_alphas))
        #print("Stopping Criteria {}".format(lr_val*t))
        #print("************")
        if fx_diff >= lr_val*t:
            break
        lr_val = tau*lr_val
    
    #print("LR Val {}".format(lr_val))

    return lr_val

def optim_boyd_dc(C, H, tolerance=0.5, iters=200, tau=0.95, control_factor=0.5):
    dim = C.shape[0]
    alphas_imo = 1
    nonzero_cols = np.nonzero(np.any(symmetrize_from_vector(H, dim) != 0, axis=0))[0]
    C_reduced = C[nonzero_cols, :][:, nonzero_cols]
    H_reduced = symmetrize_from_vector(H, dim)[nonzero_cols, :][:, nonzero_cols]
    alpha_best = 1
    likelihood_best = np.inf
    for it in range(iters):
        jacobian, hessian = calc_jacobian_hessian_dc(x=alphas_imo, H=H, C=C_reduced, dim=dim)
        inv_hessian = 1/hessian
        newton_direction = -inv_hessian*jacobian
        lambd = np.power(newton_direction*hessian*newton_direction, 0.5)
        hhat = 1
        if lambd > 0.5:
            #print("Lambda {}".format(lambd))
            # do line search
            local_slope = jacobian*newton_direction
            hhat = 1
            hhat = perform_line_search_dc(local_slope=local_slope,  
                                       newton_direction=newton_direction,
                                       alpha=alphas_imo,
                                       H=H_reduced, C=C_reduced, dim=dim, tau=tau, control_factor=control_factor
                                      )
        elif lambd < 1e-8:
            # termination condition for newton's mathod
            break
        #print('***********')
        #print("ITER {}".format(it))
        #print("Newton Direction: ", newton_direction)
        #print("Lambda ", lambd)
        new_alphas = alphas_imo + hhat*newton_direction
        likelihood_curr = boyd_likelihood_dc(x=alphas_imo, H=H_reduced, C=C_reduced, dim=C_reduced.shape[0])
        likelihood_new = boyd_likelihood_dc(x=new_alphas, H=H_reduced, C=C_reduced, dim=C_reduced.shape[0])
        likelihood_diff = likelihood_curr - likelihood_new
        if likelihood_new < likelihood_best:
            likelihood_best = likelihood_new
            alpha_best = new_alphas
        # print("Iter {} Likelihood Diff {}".format(it, likelihood_curr - likelihood_new))
        # if likelihood_diff < 0.0:
        #     print("NEGATIVE LIKELIHOOD UPDATE")
        #     pdb.set_trace()
        alphas_imo = copy.copy(new_alphas)
        #print("New Alphas", new_alphas)
        #print('***********')
        #print()
    #exit()
    #print()
    #print("################")
    #print("ALPHAS {}".format(alphas_imo))
    #print("################")
    if (alpha_best <= 0.0):
        print("ALPHAS OUT OF BOUNDS")
        exit(1)
    return alpha_best # scalar return value



#@jit(nopython=True)
# def perform_line_search(alphas, H_s, C, dim):
#     jac_wrapper = JacWrapper(H_s=H_s, C=C, dim=dim)
#     obj_fn = jac_wrapper.boyd_likelihood
#     grad_fn = jac_wrapper.calc_jacobian
#     start_point = alphas
#     search_gradient = grad_fn(alphas)
#     res = line_search(obj_fn, grad_fn, start_point, search_gradient)
#     print(res)
#     exit()

def calculate_local_slope(newton_direction, jacobian):
    return np.dot(jacobian, newton_direction)

def perform_line_search(local_slope, newton_direction, alphas, H_s, C, dim, tau, control_factor):
    t = -control_factor*local_slope
    #print("t {}".format(t))
    max_iters = 200
    # calculate maximum step size heuristic
    new_alph_temp = alphas + 1.0*newton_direction
    lr_val = 1.0
    if not (new_alph_temp <= 0).any():
        lr_val = 1.0
    else:
        while (alphas + lr_val*newton_direction <= 1e-8).any():
            lr_val = lr_val*0.95
        # alph_diff = alphas - new_alph_temp
        # min_val = np.min(new_alph_temp) # most negative value
        # min_idx = np.argmin(new_alph_temp)
        # # TODO FIX THIS LINE
        # """
        # INCOMPLETE
        # """
        # lr_val = -(alphas[min_idx]/newton_direction[min_idx])*0.5
        # """
        # """
    #print("\nLINE SEARCH\n")
    for j in range(max_iters):
        f_x = boyd_likelihood(x=alphas, H_s=H_s, C=C, dim=dim)
        new_alphas = alphas+lr_val*newton_direction
        f_x_update = boyd_likelihood(x=new_alphas, H_s=H_s, C=C, dim=dim)
        fx_diff = f_x - f_x_update
        #print("***********")
        #print("ITER {}".format(j))
        #print("FX Diff {} Orig {} New {}".format(fx_diff, f_x, f_x_update))
        #print("New Alphas {}".format(new_alphas))
        #print("Stopping Criteria {}".format(lr_val*t))
        #print("************")
        if fx_diff >= lr_val*t:
            break
        lr_val = tau*lr_val
    
    #print("LR Val {}".format(lr_val))

    return lr_val




def optim_boyd(C, H_s, tolerance=0.5, iters=10, tau=0.95, control_factor=0.5):
    dim = C.shape[0]
    M = H_s.shape[0]
    alphas_imo = np.ones(M)
    for it in range(iters):
        jacobian = calc_jacobian(x=alphas_imo, H_s=H_s, C=C, dim=dim)
        hessian = calc_hessian(alphas=alphas_imo, H_s=H_s, dim=dim)
        inv_hessian = inv(hessian)
        newton_direction = -inv_hessian.dot(jacobian)
        lambd = np.power(newton_direction.dot(hessian).dot(newton_direction), 0.5)
        hhat = 1
        if lambd > 0.5:
            #print("Lambda {}".format(lambd))
            # do line search
            local_slope = calculate_local_slope(newton_direction, jacobian)
            hhat = 1
            hhat = perform_line_search(local_slope=local_slope,  
                                       newton_direction=newton_direction,
                                       alphas=alphas_imo,
                                       H_s=H_s, C=C, dim=dim, tau=tau, control_factor=control_factor
                                      )
        elif lambd < 1e-5:
            # termination condition for newton's mathod
            break
        #print('***********')
        #print("ITER {}".format(it))
        #print("Newton Direction: ", newton_direction)
        #print("Lambda ", lambd)
        new_alphas = alphas_imo + hhat*newton_direction
        likelihood_curr = boyd_likelihood(x=alphas_imo, H_s=H_s, C=C, dim=dim)
        likelihood_new = boyd_likelihood(x=new_alphas, H_s=H_s, C=C, dim=dim)
        likelihood_diff = likelihood_curr - likelihood_new
        # print("Iter {} Likelihood Diff {}".format(it, likelihood_curr - likelihood_new))
        # if likelihood_diff < 0.0:
        #     print("NEGATIVE LIKELIHOOD UPDATE")
        #     pdb.set_trace()
        alphas_imo = new_alphas.copy()
        #print("New Alphas", new_alphas)
        #print('***********')
        #print()
    #exit()
    #print()
    #print("################")
    #print("ALPHAS {}".format(alphas_imo))
    #print("################")
    if (alphas_imo <= 0.0).any():
        print("ALPHAS OUT OF BOUNDS")
        exit(1)
    return alphas_imo

@jit(nopython=True)
def calc_psi_hat(alphas, H_s, dim):
    P = dim
    #psi_hat = sum([alphas[i]*H_s[i] for i in range(H_s.shape[0])])
    psi_hat = np.sum(np.expand_dims(alphas, 1)*H_s, 0)
    psi_hat = symmetrize_from_vector(psi_hat, P)

    return psi_hat

def calc_matrices_precision(psi_hat, H_s, C):
    dim = C.shape[0]
    H_dim = H_s.shape[0]
    A = np.zeros((H_dim, H_dim))
    B = np.zeros((H_dim,))
    inv_psi_hat = inv(psi_hat) # calculate psi_hat inverse
    for g in range(H_dim):
        H_g = symmetrize_from_vector(H_s[g], dim)
        for h in range(H_dim):
            H_h = symmetrize_from_vector(H_s[h], dim)
            mult = ((inv_psi_hat.dot(H_g)).dot(inv_psi_hat)).dot(H_h)
            lhs = np.trace(mult)
            A[g, h] = lhs
        #print("PSI_HAT PROJ", np.trace(inv_psi_hat.dot(H_g)))
        #print()
        #print("C PROJ", np.trace(C.dot(H_g)))
        #print()
        rhs = np.trace(inv_psi_hat.dot(H_g)) - np.trace(C.dot(H_g))
        B[g] = rhs
        #print("H_g", H_s[g])
        #print()
    #exit()
    return A, B

"""
Single likelihood ratio tests

Modify the iterative solution function
"""
def iterative_soln_precision_single(coeffs_zero, H_s, C, dim, modify_index=0, iters=5):
    s_imo = coeffs_zero
    s_i = s_imo.copy()
    H_dim = H_s.shape[0]
    for it in range(iters):
        #psi_hat = (s_imo.reshape(-1, 1, 1)*H_s).sum(0) # calculate psi_hat
        psi_hat = np.zeros(int((dim*(dim+1))/2))
        for i in range(H_s.shape[0]):
            psi_hat = psi_hat + s_imo[i]*H_s[i]
        psi_hat = symmetrize_from_vector(psi_hat, dim)
        A, B = calc_matrices_precision(psi_hat, H_s, C)
        #print("DIAG", np.diag(A))
        #print("A", A)
        #print("EV", np.linalg.eig(A)[0])
        t_i = inv(A).dot(B)
        s_i[modify_index] = s_imo[modify_index] + t_i[modify_index]
        s_imo = s_i
        
    return s_imo

def iterative_soln_precision(coeffs_zero, H_s, C, iters=5):
    s_imo = coeffs_zero
    s_i = s_imo.copy()
    H_dim = H_s.shape[0]
    dim = C.shape[0]
    for it in range(iters):
        #psi_hat = (s_imo.reshape(-1, 1, 1)*H_s).sum(0) # calculate psi_hat
        psi_hat = np.zeros(int((dim*(dim+1))/2))
        for i in range(H_s.shape[0]):
            psi_hat = psi_hat + s_imo[i]*H_s[i]
        psi_hat = symmetrize_from_vector(psi_hat, dim)
        A, B = calc_matrices_precision(psi_hat, H_s, C)
        # try:
        #     
        # except:
        #     pdb.set_trace()
        #t_i = inv(A).dot(B)
        t_i = np.linalg.solve(A, B)
        #print("STEP SIZE", t_i)
        #print("ALPHAS", s_imo)
        #print()
        s_i = s_imo + t_i
        s_imo = s_i
        
    return s_imo

def coord_ascent(coeffs_zero, C, H_s, modify_index=0, iters=10, beta=1e-2):
    H_dim = H_s.shape[0]
    C_dim = C.shape[0]
    new_alphas = coeffs_zero.copy()
    alpha_log = {}
    for it in range(iters):
        iter_log = {}
        H_i = symmetrize_from_vector(H_s[modify_index], C_dim)
        psi_hat = np.zeros(int((C_dim*(C_dim+1))/2))
        for i in range(H_s.shape[0]):
            psi_hat = psi_hat + new_alphas[i]*H_s[i]
        psi_hat = symmetrize_from_vector(psi_hat, C_dim)
        if not is_pos_def(psi_hat):
            pdb.set_trace()
        partial_deriv = np.trace(inv(psi_hat).dot(H_i)) - np.trace(C.dot(H_i))
        alpha_tpo = new_alphas[modify_index] + beta*partial_deriv
        new_alphas[modify_index] = alpha_tpo

        iter_log['step_size'] = beta*partial_deriv
        iter_log['deriv'] = partial_deriv
        iter_log['alphas'] = new_alphas.copy()
        iter_log['logdet'] = np.log(np.linalg.det(psi_hat))
        iter_log['det'] = np.linalg.det(psi_hat)
        iter_log['eigvals'] = np.linalg.eig(psi_hat)[0]
        alpha_log[it] = iter_log
    return new_alphas

def unbiased_init_precision(C, H_s):
    H_dim = H_s.shape[0]
    A = np.zeros((H_dim, H_dim))
    B = np.zeros((H_dim, ))
    C_dim = C.shape[0]
    for g in range(H_dim):
        for h in range(H_dim):
            H_g = symmetrize_from_vector(H_s[g], C_dim)
            H_h = symmetrize_from_vector(H_s[h], C_dim)
            A[g, h] = np.trace(H_g.dot(H_h))
        B[g] = np.trace(inv(C).dot(H_g))
    return np.linalg.solve(A, B)

def unbiased_init_precision_single(coeffs_zero, C, H_s, modify_index=0):
    H_dim = H_s.shape[0]
    C_dim = C.shape[0]
    A = np.zeros((H_dim, ))
    B = np.zeros((H_dim, ))
    H_curr = symmetrize_from_vector(H_s[modify_index], C_dim)
    # populate A vector
    for i in range(H_dim):
        H_i = symmetrize_from_vector(H_s[i], C_dim)
        A[i] = np.trace(H_curr.dot(H_i))
    
    # populate B vector
    for i in range(H_dim):
        H_i = symmetrize_from_vector(H_s[i], C_dim)
        lhs = np.trace(inv(C).dot(H_i))
        rhs = 0
        for j in range(H_dim):
            if j != modify_index:
                rhs += coeffs_zero[j] * np.trace(H_curr.dot(H_i))
        B[i] = lhs - rhs
    
    A = np.expand_dims(A, 1)
    #print(A, B)
    soln = np.linalg.lstsq(A, B)[0]
    print("SOLN", soln)
    new_alphas = coeffs_zero.copy()
    new_alphas[modify_index] = soln

    return new_alphas

def unbiased_init_precision_single_alt(coeffs_zero, C, H_s, modify_index=0):
    H_dim = H_s.shape[0]
    C_dim = C.shape[0]
    H_curr = symmetrize_from_vector(H_s[modify_index], C_dim)
    lhs = np.trace(inv(C).dot(H_curr))
    rhs = 0
    for i in range(H_dim):
        if i != modify_index:
            H_i = symmetrize_from_vector(H_s[i], C_dim)
            rhs += coeffs_zero[i] * np.trace(H_i.dot(H_curr))
    soln = (lhs-rhs)/np.trace(H_curr.dot(H_curr))

    new_alphas = coeffs_zero.copy()
    new_alphas[modify_index] = soln
    print("SOLN", soln)
    return new_alphas


class CVXProblemCluster(object):
    def __init__(self, dim, full_dim):
        super().__init__()
        self.dim = dim
        self.full_dim = full_dim

    def create_single_optim_problem(self, H_s, optim_idx=0):
        self.single_alpha = cp.Variable()
        self.C = cp.Parameter((self.dim, self.dim))
        psi_hat = cp.Variable((self.dim, self.dim))
        self.H_i = symmetrize_from_vector(H_s[optim_idx], self.full_dim)
        self.H_i_reduced = self.H_i[:, ~np.all(self.H_i == 0, axis=0)]
        self.H_i_reduced = self.H_i_reduced[~np.all(self.H_i_reduced == 0, axis=1), :]
        constr1 = ((psi_hat - np.eye(self.dim)*1e-6) >> 0)
        constr2 = (psi_hat == psi_hat.T)
        self.objective = cp.Maximize(cp.log_det(psi_hat) - cp.trace(psi_hat@self.C))
        self.problem = cp.Problem(self.objective, 
                                 [constr1, constr2, psi_hat==self.H_i_reduced*self.single_alpha])
        assert self.problem.is_dcp(), "Not DCP"
        assert self.problem.is_dpp(), "Not DPP"

def create_all_optim_problems_cluster(H_s, dim):
    prob_dict = {}
    
    for i in range(H_s.shape[0]):
        curr_H = symmetrize_from_vector(H_s[i], dim)
        curr_H = curr_H[:, ~np.all(curr_H == 0, axis=0)]
        curr_H = curr_H[~np.all(curr_H == 0, axis=1), :]
        assert curr_H.shape[0] == curr_H.shape[1]
        curr_prob = CVXProblemCluster(dim=curr_H.shape[0], full_dim=dim)
        curr_prob.create_single_optim_problem(H_s=H_s, optim_idx=i)
        prob_dict[i] = curr_prob
    return prob_dict

def solve_optim_single_cluster(optim_idx, curr_alphas, curr_C, prob_dict):
    curr_prob = prob_dict[optim_idx]
    curr_prob.C.value = curr_C
    curr_prob.problem.solve(
        solver=cp.SCS, 
        verbose=False, 
        warm_start=True,
        scale=1.0,
        adaptive_scale=True,
        max_iters=int(1e4)
        )
    new_alphas = curr_alphas.copy()
    new_alphas[optim_idx] = curr_prob.single_alpha.value
    return new_alphas

    
class CVXProblem(object):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def create_single_optim_problem(self, H_s, optim_idx=0):
        self.single_alpha = cp.Variable()
        self.other_alphas = cp.Parameter(H_s.shape[0]-1)
        y = cp.Variable((self.dim, self.dim))
        psi_hat = cp.Variable((self.dim, self.dim))
        temp_psi_hat = cp.Variable((self.dim, self.dim))
        self.C = cp.Parameter((self.dim, self.dim))
        list_vals = []
        alph_counter = 0
        for i in range(H_s.shape[0]):
            if i != optim_idx:
                list_vals.append(self.other_alphas[alph_counter]*symmetrize_from_vector(H_s[i], self.dim))
                alph_counter += 1
        constr1 = ((psi_hat - np.eye(self.dim)*1e-6) >> 0)
        constr2 = (psi_hat == psi_hat.T)
        self.objective = cp.Maximize(cp.log_det(psi_hat) - cp.trace(psi_hat@self.C))
        self.problem = cp.Problem(self.objective, 
                                 [constr1, constr2, psi_hat==temp_psi_hat + y, 
                                 y==self.single_alpha*symmetrize_from_vector(H_s[optim_idx], self.dim),  
                                 temp_psi_hat==cp.sum(list_vals)])
        assert self.problem.is_dcp(), "Not DCP"
        assert self.problem.is_dpp(), "Not DPP"
        
    def create_global_optim_problem(self, H_s):
        self.global_alphas = cp.Variable(shape=(H_s.shape[0], ))
        psi_hat = cp.Variable((self.dim, self.dim))
        self.C = cp.Parameter((self.dim, self.dim))
        list_vals = [self.global_alphas[i]*symmetrize_from_vector(H_s[i], self.dim) for i in range(H_s.shape[0])]
        constr1 = (psi_hat >> 0)
        constr2 = (psi_hat == psi_hat.T)
        self.objective = cp.Maximize(cp.log_det(psi_hat) - cp.trace(psi_hat@self.C))
        self.problem = cp.Problem(self.objective, 
                                 [constr1, constr2, psi_hat==cp.sum(list_vals)])
        
def create_all_optim_problems(H_s, dim):
    prob_dict = {}
    
    for i in range(H_s.shape[0]):
        curr_prob = CVXProblem(dim=dim)
        curr_prob.create_single_optim_problem(H_s=H_s, optim_idx=i)
        prob_dict[i] = curr_prob
    return prob_dict

def create_global_problem(H_s, dim):
    g_prob = CVXProblem(dim=dim)
    g_prob.create_global_optim_problem(H_s=H_s)
    
    return g_prob

def solve_optim_single(optim_idx, curr_alphas, curr_C, prob_dict):
    other_alphas = np.delete(curr_alphas, optim_idx)
    curr_prob = prob_dict[optim_idx]
    curr_prob.other_alphas.value = other_alphas
    curr_prob.C.value = curr_C
    curr_prob.problem.solve(
        solver=cp.SCS, 
        verbose=False, 
        warm_start=True,
        scale=1.0,
        adaptive_scale=True,
        max_iters=int(1e4)
        )
    new_alphas = curr_alphas.copy()
    new_alphas[optim_idx] = curr_prob.single_alpha.value
    return new_alphas

def solve_optim_global(g_prob, curr_C):
    g_prob.C.value = curr_C
    g_prob.problem.solve(solver=cp.SCS, verbose=False, warm_start=True)
    
    return g_prob.global_alphas.value

def optimize_coeffs(H_s, C, lam=1e-2):
    M = H_s.shape[0]
    dim = C.shape[0]
    alphas = cp.Variable(shape=(H_s.shape[0], ))
    psi_hat = sum([alphas[i]*symmetrize_from_vector_alt(H_s[i], dim) for i in range(M)])
    l1_penalty = sum([cp.abs(psi_hat[i, j])
                for i in range(dim)
                for j in range(dim) if i != j])
    objective = cp.Maximize(cp.log_det(psi_hat) - cp.trace(psi_hat@C)) #- lam*l1_penalty)
    constr1 = (psi_hat >> 0)
    constr2 = (psi_hat == psi_hat.T)
    constraints = [constr1, constr2]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)
    if problem.status != cp.OPTIMAL:
        raise Exception('CVXPY Error')
        
    return alphas.value

def optimize_single_coeff(alphas, H_s, C, coeff_idx=0, lam=1e-2):
    M = H_s.shape[0]
    dim = C.shape[0]
    single_alpha = cp.Variable()
    psi_hat = np.zeros((dim, dim))
    temp_psi_hat = np.zeros((dim, dim))
    for i in range(M):
        if i != coeff_idx:
            temp_psi_hat += alphas[i]*symmetrize_from_vector_alt(H_s[i], dim)
    psi_hat = single_alpha * symmetrize_from_vector_alt(H_s[coeff_idx], dim) + temp_psi_hat
#     for i in range(M):
#         if i == coeff_idx:
#             psi_hat += cp.multiply(single_alpha, symmetrize_from_vector_alt(H_s[i], dim))
#         else:
#             psi_hat += alphas[i]*symmetrize_from_vector_alt(H_s[i], dim)
    l1_penalty = sum([cp.abs(psi_hat[i, j])
                for i in range(dim)
                for j in range(dim) if i != j])
    objective = cp.Maximize(cp.log_det(psi_hat) - cp.trace(psi_hat@C))# - lam*l1_penalty)
    constr1 = (psi_hat >> 0)
    constr2 = (psi_hat == psi_hat.T)
    constraints = [constr1, constr2]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, 
                  verbose=False, 
                  eps_rel=1e-5, 
                  eps_infeas=1e-7,
                  alpha=1.0
                 )
#     if problem.status != cp.OPTIMAL:
#         raise Exception('CVXPY Error')
    
    new_alphas = alphas.copy()
    new_alphas[coeff_idx] = single_alpha.value
    return new_alphas

@jit(nopython=True)
def l1_penalty_subderiv(alphas, H, i):
    l1_pen_sum = 0
    dim = H.shape[0]
    for j in range(dim):
        for k in range(dim):
            if j != k:
                if H[j,k] == 0 or alphas[i] == 0:
                    continue
                else:
                    term = H[j,k]
                    l1_pen_sum += term*np.sign(term)*np.sign(alphas[i])
    return l1_pen_sum

@jit(nopython=True)
def gradient_step(alphas, H_s, C, lam=1e-2, beta=1e-2, t=2.0):
    M = H_s.shape[0]
    dim = C.shape[0]
    alphas_new = np.zeros(alphas.shape)
    psi_hat = np.zeros(int((dim*(dim+1))/2))
    for i in range(H_s.shape[0]):
        psi_hat = psi_hat + alphas[i]*H_s[i]
    psi_hat = symmetrize_from_vector(psi_hat, dim)
    inv_psi_hat = inv(psi_hat)
    for i in range(M):
        first = np.trace(symmetrize_from_vector(H_s[i], dim)@C)
        second = np.trace(inv_psi_hat@symmetrize_from_vector(H_s[i], dim))
        log_barrier = (1/t)*second
        l1_penalty = l1_penalty_subderiv(alphas, symmetrize_from_vector(H_s[i], dim), i)
        deriv = first - second #- log_barrier #+ lam*l1_penalty
        alphas_new[i] = alphas[i] - beta*deriv

    return alphas_new

@jit(nopython=True)
def gradient_step_single(alphas, H_s, C, lam=1e-2, beta=1e-2, t=2.0, optim_indx=0):
    M = H_s.shape[0]
    dim = C.shape[0]
    alphas_new = alphas.copy()
    psi_hat = np.zeros(int((dim*(dim+1))/2))
    for i in range(H_s.shape[0]):
        psi_hat = psi_hat + alphas[i]*H_s[i]
    psi_hat = symmetrize_from_vector(psi_hat, dim)
    inv_psi_hat = inv(psi_hat)
    first = np.trace(symmetrize_from_vector(H_s[optim_indx], dim)@C)
    second = np.trace(inv_psi_hat@symmetrize_from_vector(H_s[optim_indx], dim))
    log_barrier = (1/t)*second
    l1_penalty = l1_penalty_subderiv(alphas, symmetrize_from_vector(H_s[optim_indx], dim), optim_indx)
    deriv = first - second #- log_barrier #+ lam*l1_penalty
    alphas_new[optim_indx] = alphas[optim_indx] - beta*deriv

    return alphas_new

@jit(nopython=True)
def optimize_coeffs_first_order(H_s, C, lam=1e-2, beta=1e-2, iters=200, include_l1=True, t=2.0):
    #print('*********\n\n')
    M = H_s.shape[0]
    alphas_imo = np.ones(M)
    best_likelihood = lasso_likelihood(alphas_imo, H_s, C, lam=lam, include_l1=include_l1)
    best_coeffs = alphas_imo.copy()
    for it in range(iters):
        alphas_imo = gradient_step(alphas_imo, H_s, C, lam=lam, beta=beta, t=t)
        likelihood = lasso_likelihood(alphas_imo, H_s, C, lam=lam, include_l1=include_l1)
        if likelihood > best_likelihood:
            best_likelihood = likelihood
            best_coeffs = alphas_imo.copy()
        # schedule t
        t = t*1.4

    #best_coeffs = alphas_imo.copy()
    return best_coeffs

@jit(nopython=True)
def optimize_coeffs_first_order_single(alphas, H_s, C, lam=1e-2, beta=1e-2, iters=200, optim_indx=0, include_l1=True, t=2.0):
    M = H_s.shape[0]
    alphas_imo = alphas.copy()
    best_likelihood = lasso_likelihood(alphas_imo, H_s, C, lam=lam, include_l1=include_l1)
    best_coeffs = alphas_imo.copy()
    for it in range(iters):
        alphas_imo = gradient_step_single(alphas_imo, H_s, C, lam=lam, beta=beta, t=t, optim_indx=optim_indx)
        likelihood = lasso_likelihood(alphas_imo, H_s, C, lam=lam, include_l1=include_l1)
        if likelihood > best_likelihood:
            best_likelihood = likelihood
            best_coeffs = alphas_imo.copy()
        # schedule t
        t = t*1.4
    
    #best_coeffs = alphas_imo.copy()
    return best_coeffs



