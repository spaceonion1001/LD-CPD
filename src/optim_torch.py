import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def cost_func(psi_hat, C, lam=1e-2, t=20):
    dim = C.shape[0]
    first = torch.trace(C@psi_hat)
    second = torch.logdet(psi_hat)
    l1_penalty = sum([torch.abs(psi_hat[i, j])
                for i in range(dim)
                for j in range(dim) if i != j])
    last = (1/t)*torch.logdet(psi_hat)
    return first - second + lam*l1_penalty - last

def optimize_coeffs_torch(H_s, C, lam=1e-2, iters=50):
    M = H_s.shape[0]
    H_s_torch = torch.FloatTensor(H_s)
    alphas = nn.Parameter(torch.ones(M)).float()
    optimizer = optim.Adam([alphas], lr=1e-1, betas=(0.9, 0.999))
    for it in range(iters):
        optimizer.zero_grad()
        psi_hat = sum([alphas[i]*H_s_torch[i] for i in range(M)])
        loss = cost_func(psi_hat, torch.FloatTensor(C), lam=1e-2, t=20)
        loss.backward()
        optimizer.step()
    return alphas.detach().numpy()

def optimize_single_coeff_torch(alphas, H_s, C, coeff_idx=0, lam=1e-2, iters=50):
    M = H_s.shape[0]
    H_s_torch = torch.FloatTensor(H_s)
    alphas_torch = torch.FloatTensor(alphas)
    alpha = nn.Parameter(torch.ones(1)).float()
    optimizer = optim.Adam([alpha], lr=1e-1, betas=(0.9, 0.999))
    for it in range(iters):
        optimizer.zero_grad()
        psi_hat = torch.zeros(C.shape)
        for i in range(M):
            if i == coeff_idx:
                psi_hat += alpha*H_s_torch[i]
            else:
                psi_hat += alphas_torch[i]*H_s_torch[i]
        loss = cost_func(psi_hat, torch.FloatTensor(C), lam=1e-2, t=20)
        loss.backward()
        optimizer.step()
    new_alphas = alphas_torch.numpy().copy()
    new_alphas[coeff_idx] = alpha.detach().numpy()
    return new_alphas


###################
# def optimize_single_coeff(alphas, H_s, C, coeff_idx=0, lam=1e-2):
#     M = H_s.shape[0]
#     dim = C.shape[0]
#     single_alpha = cp.Variable()
#     psi_hat = np.zeros(C.shape)
#     for i in range(M):
#         if i == coeff_idx:
#             psi_hat += single_alpha*H_s[i]
#         else:
#             psi_hat += alphas[i]*H_s[i]
#     l1_penalty = sum([cp.abs(psi_hat[i, j])
#                 for i in range(dim)
#                 for j in range(dim) if i != j])
#     objective = cp.Maximize(cp.log_det(psi_hat) - cp.trace(psi_hat@C) - lam*l1_penalty)
#     constr1 = (psi_hat >> 0)
#     constr2 = (psi_hat == psi_hat.T)
#     constraints = [constr1, constr2]
#     problem = cp.Problem(objective, constraints)
#     problem.solve(solver=cp.SCS)
#     if problem.status != cp.OPTIMAL:
#         raise Exception('CVXPY Error')
    
#     new_alphas = alphas.copy()
#     new_alphas[coeff_idx] = single_alpha.value
#     return new_alphas