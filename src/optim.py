import numpy as np
from numpy.linalg import inv
import cvxpy as cp
from utils import lasso_likelihood, vectorize_matrix, symmetrize_from_vector


def optimize_coeffs(H_s, C, lam=1e-2):
    M = H_s.shape[0]
    dim = C.shape[0]
    alphas = cp.Variable(shape=(H_s.shape[0], ))
    psi_hat = sum([alphas[i]*H_s[i] for i in range(M)])
    psi_hat = symmetrize_from_vector(psi_hat, dim)
    l1_penalty = sum([cp.abs(psi_hat[i, j])
                for i in range(dim)
                for j in range(dim) if i != j])
    objective = cp.Maximize(cp.log_det(psi_hat) - cp.trace(psi_hat@C) - lam*l1_penalty)
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
    psi_hat = np.zeros(H_s[0].shape)
    for i in range(M):
        if i == coeff_idx:
            psi_hat += single_alpha*H_s[i]
        else:
            psi_hat += alphas[i]*H_s[i]
    psi_hat = symmetrize_from_vector(psi_hat, dim)
    l1_penalty = sum([cp.abs(psi_hat[i, j])
                for i in range(dim)
                for j in range(dim) if i != j])
    objective = cp.Maximize(cp.log_det(psi_hat) - cp.trace(psi_hat@C) - lam*l1_penalty)
    constr1 = (psi_hat >> 0)
    constr2 = (psi_hat == psi_hat.T)
    constraints = [constr1, constr2]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS)
    if problem.status != cp.OPTIMAL:
        raise Exception('CVXPY Error')
    
    new_alphas = alphas.copy()
    new_alphas[coeff_idx] = single_alpha.value
    return new_alphas

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

def gradient_step(alphas, H_s, C, lam=1e-2, beta=1e-2, t=3):
    M = H_s.shape[0]
    dim = C.shape[0]
    alphas_new = np.zeros(alphas.shape)
    psi_hat = sum([alphas[i]*H_s[i] for i in range(H_s.shape[0])])
    psi_hat = symmetrize_from_vector(psi_hat, dim)
    inv_psi_hat = inv(psi_hat)
    for i in range(M):
        first = np.trace(symmetrize_from_vector(H_s[i], dim)@C)
        second = np.trace(inv_psi_hat@symmetrize_from_vector(H_s[i], dim))
        log_barrier = (1/t)*second
        l1_penalty = l1_penalty_subderiv(alphas, symmetrize_from_vector(H_s[i], dim), i)
        deriv = first - second - log_barrier + lam*l1_penalty
        alphas_new[i] = alphas[i] - beta*deriv

    return alphas_new

def gradient_step_single(alphas, H_s, C, lam=1e-2, beta=1e-2, t=3, optim_indx=0):
    M = H_s.shape[0]
    dim = C.shape[0]
    alphas_new = alphas.copy()
    psi_hat = sum([alphas[i]*H_s[i] for i in range(H_s.shape[0])])
    psi_hat = symmetrize_from_vector(psi_hat, dim)
    inv_psi_hat = inv(psi_hat)
    first = np.trace(symmetrize_from_vector(H_s[optim_indx], dim)@C)
    second = np.trace(inv_psi_hat@symmetrize_from_vector(H_s[optim_indx], dim))
    log_barrier = (1/t)*second
    l1_penalty = l1_penalty_subderiv(alphas, symmetrize_from_vector(H_s[optim_indx], dim), optim_indx)
    deriv = first - second - log_barrier + lam*l1_penalty
    alphas_new[optim_indx] = alphas[optim_indx] - beta*deriv

    return alphas_new

def optimize_coeffs_first_order(H_s, C, lam=1e-2, beta=1e-2, iters=200, include_l1=True):
    M = H_s.shape[0]
    alphas_imo = np.ones(M)
    best_likelihood = lasso_likelihood(alphas_imo, H_s, C, lam=lam, include_l1=include_l1)
    best_coeffs = alphas_imo.copy()
    for it in range(iters):
        alphas_imo = gradient_step(alphas_imo, H_s, C, lam=lam, beta=beta)
        likelihood = lasso_likelihood(alphas_imo, H_s, C, lam=lam, include_l1=include_l1)
        if likelihood > best_likelihood:
            best_likelihood = likelihood
            best_coeffs = alphas_imo.copy()
    
    #best_coeffs = alphas_imo.copy()
    return best_coeffs

def optimize_coeffs_first_order_single(alphas, H_s, C, lam=1e-2, beta=1e-2, iters=200, optim_indx=0, include_l1=True):
    M = H_s.shape[0]
    alphas_imo = alphas.copy()
    best_likelihood = lasso_likelihood(alphas_imo, H_s, C, lam=lam, include_l1=include_l1)
    best_coeffs = alphas_imo.copy()
    for it in range(iters):
        alphas_imo = gradient_step_single(alphas_imo, H_s, C, lam=lam, beta=beta, optim_indx=optim_indx)
        likelihood = lasso_likelihood(alphas_imo, H_s, C, lam=lam, include_l1=include_l1)
        if likelihood > best_likelihood:
            best_likelihood = likelihood
            best_coeffs = alphas_imo.copy()
    
    #best_coeffs = alphas_imo.copy()
    return best_coeffs



