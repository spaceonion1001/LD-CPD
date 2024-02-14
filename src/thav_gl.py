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


from statsmodels.tsa.seasonal import STL

from utils import is_symmetric, is_pos_def, vectorize_matrix, symmetrize_from_vector


import warnings
warnings.filterwarnings('ignore')  # <- remember to comment this if something breaks and you get confused

def thav_gl_fn(data_train, lambda_search, C=0.5, threshold=0.5):
    j = len(lambda_search) - 2
    r = lambda_search[j]
    r_hat = lambda_search[0]
    #data_cov = np.corrcoef(data_train)
    data_cov = np.cov(data_train)
    while r > lambda_search[0]:
        #glasso_r = GraphicalLasso(max_iter=1500, alpha=r, tol=1e-4, verbose=False).fit(data_train.T)
        try:
            glasso_r = GraphicalLasso(max_iter=1500, alpha=r, tol=1e-4, verbose=False, covariance='precomputed', eps=1e-3).fit(data_cov)
        except FloatingPointError:
            print("ILL CONDITIONED")
        theta_hat_r = glasso_r.precision_
        j_prime = len(lambda_search) - 1
        r_prime = lambda_search[j_prime]
        while r_prime > r:
            #glasso_rprime = GraphicalLasso(max_iter=1500, alpha=r_prime, tol=1e-4, verbose=False).fit(data_train.T)
            try:
                glasso_rprime = GraphicalLasso(max_iter=1500, alpha=r_prime, tol=1e-4, verbose=False, covariance='precomputed', eps=1e-3).fit(data_cov)
            except FloatingPointError:
                print("ILL CONDITIONED")
            theta_hat_rprime = glasso_rprime.precision_
            if np.linalg.norm(theta_hat_r-theta_hat_rprime, ord=np.inf) > C*(r+r_prime):
                r_hat = lambda_search[j+1]
                break
            else:
                j_prime = j_prime - 1
                r_prime = lambda_search[j_prime]
        j = j - 1
        r = lambda_search[j]
    #glasso_rhat = GraphicalLasso(max_iter=1500, alpha=r_hat, tol=1e-4, verbose=False).fit(data_train.T)
    glasso_rhat = GraphicalLasso(max_iter=1500, alpha=r_hat, tol=1e-4, verbose=False, covariance='precomputed', eps=1e-3).fit(data_cov)
    theta_hat_rhat = glasso_rhat.precision_
    a_v = theta_hat_rhat
    t = threshold*C*r_hat
    a_v[np.abs(a_v) < t] = 0.0

    return a_v, r_hat