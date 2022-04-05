import numpy as np
import os
from datetime import datetime

from sklearn.preprocessing import StandardScaler


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
    print('Loading Alaska Data...')
    data = np.loadtxt(os.path.join(args.data_path, 'hj_andrews_resid.csv'))
    return data

def scale_data(data):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    return data_scaled

def create_fig_dir(fig_path):
    today = datetime.now()

    new_dir_path = os.path.join(fig_path, today.strftime('%Y%m%d%H'))
    if not os.path.isdir(new_dir_path):
        os.mkdir(new_dir_path)

    return new_dir_path

    