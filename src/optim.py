import numpy as np
import cvxpy as cp


def optimize_coeffs(H_s, C, lam=1e-2):
    M = H_s.shape[0]
    dim = C.shape[0]
    alphas = cp.Variable(shape=(H_s.shape[0], ))
    psi_hat = sum([alphas[i]*H_s[i] for i in range(M)])
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
    psi_hat = np.zeros(C.shape)
    for i in range(M):
        if i == coeff_idx:
            psi_hat += single_alpha*H_s[i]
        else:
            psi_hat += alphas[i]*H_s[i]
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