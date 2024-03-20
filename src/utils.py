import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from numba import jit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import groupby

def is_pos_def(A):
    if is_symmetric(A):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

def is_symmetric(a, tol=1e-8):
    return np.all(np.abs(a-a.T) < tol)

def load_alaska_data(args):
    print('Loading Alaska Data...')
    data = np.loadtxt(os.path.join(args.data_path, 'dtec/alaska.csv'), delimiter=',')
    return data

def load_hjandrews_data(args):
    print('Loading HJ Andrews Data...')
    data = np.loadtxt(os.path.join(args.data_path, 'hj_andrews_resid.csv'), delimiter=',')
    random_sample = np.random.choice(np.arange(data.shape[1]), 40)

    return data[:, random_sample]

def load_holiday_farm_data(args):
    print('Loading Holiday Farm Data...')
    data = np.loadtxt(os.path.join(args.data_path, 'holiday_farm_clean.csv'), delimiter=',')
    return data

def load_tohoku_data(args):
    print('Loading Tohoku Data...')
    data = np.loadtxt(os.path.join(args.data_path, 'dtec/tk_two.csv'), delimiter=',')
    return data

def load_stock_market_data(args):
    print('Loading Stock Market Data...')
    data = np.loadtxt(os.path.join(args.data_path, 'logdiff_vals.csv'), delimiter=',')
    return data

def load_mesonet_data(args):
    print('Loading MesoNet Data...')
    data = pd.read_csv(os.path.join(args.data_path, args.data_fname), delimiter=',')
    data = data.drop('YYYYMMDDhhmm', axis=1)
    
    #data = data.loc[:,~data.columns.str.contains('GUTH', case=False)] 
    for col in data.columns:
        data.loc[data[col]>np.mean(data[col])+5*np.std(data[col]),col] = np.nan
        data.loc[data[col]<np.mean(data[col])-5*np.std(data[col]),col] = np.nan
    data = pd.DataFrame(data).interpolate('linear', axis=0).fillna(method='backfill') # interpolate
    data = data.diff(1).dropna()
    data = data.values.astype(np.float64)
    #data_df = pd.DataFrame(data).diff(1).dropna() # first differencing
    #data = data_df.values
    
    # scaler = StandardScaler()
    # data = scaler.fit_transform(data)
    # quantiles = pd.DataFrame(data).quantile(0.95).values
    # for col in range(data.shape[1]):
    #     curr_col = data[:, col]
    #     std_col = curr_col.std()
    #     #curr_col[np.abs(curr_col) >= 3*std_col] = 3*std_col
    #     curr_col[np.abs(curr_col) >= quantiles[col]] = quantiles[col]
    #     data[:, col] = curr_col.copy()
    
    
    #sns.histplot(data.values, bins=40, legend=False)
    #plt.savefig('mesonet_hist.png')
    #plt.close()

    return data

def load_mesonet_pressure_data(args):
    print('Loading Mesonet Pressure Data...')
    raw_df = pd.read_csv(os.path.join(args.data_path, args.data_fname), delimiter=',')
    raw_df = raw_df.drop('YYYYMMDDhhmm', axis=1)
    for col in raw_df.columns:
        gr = max([list(group) for _, group in groupby(raw_df.loc[:, col].values)], key=len)
        if len(gr) >= 12.0:
            raw_df = raw_df.drop(col, axis=1)
        else:
            raw_df.loc[raw_df[col]<=0.0,col] = np.nan
            # raw_df.loc[raw_df[col]>np.mean(raw_df[col])+2*np.std(raw_df[col]),col] = np.nan
            # raw_df.loc[raw_df[col]<np.mean(raw_df[col])-2*np.std(raw_df[col]),col] = np.nan
    raw_df = raw_df.interpolate('linear', axis=0).fillna(method='backfill')
    raw_df = raw_df.diff(1).dropna()
    raw_df = raw_df.values.astype(np.float64)
    
    np.random.seed(42)
    random_sampled_cols = np.random.choice(np.arange(0, raw_df.shape[1]), replace=False, size=80)
    raw_df = raw_df[:, random_sampled_cols]

    return raw_df


def load_sap_data(args):
    print('Loading SAP500 Data...')
    data = pd.read_csv(os.path.join(args.data_path, 'sap_scaled_returns.csv'), delimiter='\t')
    data = data.drop('Date', axis=1)

    return data.values



def scale_data(data, percent=1.0, end_idx=None):
    scaler = StandardScaler()
    if percent is not None:
        end_idx = int(percent*data.shape[0])
    scaler = scaler.fit(data[:end_idx])
    data_scaled = scaler.transform(data)

    return data_scaled

def vectorize_matrix(A):
    """
    Turns a symmetric matrix A into a vector from the lower triangle
    """
    indxes = np.tril_indices(A.shape[0])

    return A[indxes]

@jit(nopython=True)
def symmetrize_from_vector(a, dim):
    """
    Turns a vector of lower triangular matrix entries into symmetric matrix
    """
    A = np.zeros((dim,dim))
    ti_1, ti_2 = np.tril_indices(A.shape[0], k=0)
    for idx in range(a.shape[0]):
        val = a[idx]
        i = ti_1[idx]
        j = ti_2[idx]
        A[i,j] = val
    A = A + A.T - np.diag(np.diag(A))

    return A

def symmetrize_from_vector_alt(a, dim):
    """
    Turns a vector of lower triangular matrix entries into symmetric matrix
    """
    A = np.zeros((dim,dim))
    A[np.tril_indices(A.shape[0], k = 0)] = a
    A = A + A.T - np.diag(np.diag(A))

    return A

def difference_data(data, order=1):
    temp_df = pd.DataFrame(data)
    temp_df = temp_df.diff(order, axis=0)
    return temp_df.values[1:, :]

def create_fig_dir(fig_path):
    today = datetime.now()

    new_dir_path = os.path.join(fig_path, today.strftime('%Y%m%d%H'))
    if not os.path.isdir(new_dir_path):
        os.mkdir(new_dir_path)

    return new_dir_path

@jit(nopython=True)
def lasso_likelihood(alphas, H_s, C, lam=1e-2, include_l1=False):
    dim = C.shape[0]
    psi_hat = np.zeros(int((dim*(dim+1))/2))
    for i in range(H_s.shape[0]):
        psi_hat = psi_hat + alphas[i]*H_s[i]
    psi_hat = symmetrize_from_vector(psi_hat, dim)
    l1_penalty = 0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if i != j:
                l1_penalty += np.abs(psi_hat[i,j])
    if include_l1:
        return np.log(np.linalg.det(psi_hat)) - np.trace(psi_hat@C) - lam*l1_penalty
    
    else:
        return np.log(np.linalg.det(psi_hat)) - np.trace(psi_hat@C)

def amoc_gen(alarms, fire_point, max_time_detection):
    """
    Generate an AMOC curve.
    Arguments: 
        alarms: p-vals?
        fire_point: day of fire onset (int)
        max_time_detection: the maximum number of days to detect a fire (int)
    Returns: 
        FPR_array: false positive rates (list)
        detection_array: days to detection time (list)
    """
    thresholds = np.arange(0, 1, .01)
    FPR_array = []
    detection_array = []

    # Check FPR and days to detection for each probability threshold
    for threshold in thresholds:
        detected = False
        detection_time = 0
        false_positives = 0
        true_negatives = 0
        
        for p_value in alarms:
            
            # Two cases: false postive and true positive
            if p_value <= threshold:
                if detection_time <= fire_point: # Check for false positive
                    false_positives += 1
                
                else: # True positive: fire detected
                    detected = True
                    detection_array.append(detection_time - fire_point)
                    break
            
            else: # Two cases: false negative and true negative
                if detection_time < fire_point: # Check for true negative -- needed for FPR
                    true_negatives += 1
            detection_time += 1

        if false_positives + true_negatives != 0:
            fp_rate = (false_positives) / (false_positives + true_negatives)
            
        else:
            fp_rate = 0
            pass

        FPR_array.append(fp_rate)
        if not detected:
            detection_array.append(max_time_detection)
    
    return FPR_array, detection_array

    