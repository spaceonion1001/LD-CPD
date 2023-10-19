import numpy as np
from utils import is_symmetric, is_pos_def, vectorize_matrix, symmetrize_from_vector

def meinshausen_correction(H_s, p_vals_all, dim, log_pvals=None):
    t, M = p_vals_all.shape
    print(t, M, dim)
    p_vals_corrected = np.zeros_like(p_vals_all)
    for i in range(M):
        curr_mat = symmetrize_from_vector(H_s[i], dim)
        nonzero_cols = np.nonzero(np.any(curr_mat != 0, axis=0))[0]
        C_val = len(nonzero_cols)
        if not log_pvals:
            if C_val > 0:
                correction_factor = dim/C_val
                print(correction_factor)
                # correct p_vals for cluster at all time points independently
                p_vals_corrected[:, i] = p_vals_all[:, i]*correction_factor
            else:
                p_vals_corrected[:, i] = 1.0
        else:
            # log scale
            if C_val > 0:
                correction_factor = np.log(dim) - np.log(C_val)
                p_vals_corrected[:, i] = p_vals_all[:, i]+correction_factor
            else:
                p_vals_corrected[:, i] = 0.0
    
    return p_vals_corrected
    