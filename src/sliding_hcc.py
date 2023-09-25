import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta as beta
from scipy.stats import f as f
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from datetime import datetime
from datetime import timedelta

import seaborn as sns
import os
import glob
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import chi2, multivariate_normal

import argparse

sns.set()
sns.set_palette("viridis")

import math

### get a single point for a phase 2 control point
def phase2controlpoint(df, alpha, point_idx, win_size):
    data = df.drop(['YYYYMMDDhhmm'], axis = 1).to_numpy()

    train_data = data[point_idx - win_size:point_idx]
    #future_data = data[train_idx:]
    
    p = train_data.shape[1]
    m = train_data.shape[0]
    
    #means calculated from historical
    means = []
    for i in range(train_data.shape[1]):
        v_i = train_data[:,i].mean()
        means.append(v_i)
        
    qs = []
    m = data.shape[0]
    
    #s calculated from historical
    s = np.linalg.inv(np.cov(np.transpose(train_data)))
    
    v_point = data[point_idx] - means

    
    vt_point = np.transpose(v_point)
    
    q = v_point @ s @ vt_point 

# FOR RUNNING ON A STATIC WINDOW
#     for i in tqdm(range(data.shape[0])):
#         v_i = data[i] - means
#         #v_i = v_i
#         vt_i = np.transpose(v_i)
#         q = v_i @ s @ vt_i
#         qs.append(q)
    return q

def get_ucl(alpha, p, m):
    ALPHA = alpha
    f_ucl = f.ppf(1 - (ALPHA /2), p, m - p)
    # f_lcl = f.ppf((ALPHA /2), p, m - p)

    mpart = (p*(m+1)*(m-1)) / (m**2 - (m*p))
    ucl = mpart*f_ucl
    # lcl = mpart*f_lcl
    return ucl

def pos_argmin(times):
    argmin = -1
    amin = timedelta(days=370)
    for i in range(len(times)):
        if(times[i] > timedelta(days=0)):
            if(times[i] < amin):
                amin = times[i]
                argmin = i
    return argmin


def sliding_control(data_df, alpha, win_size, end_idx):
    
    ucl = get_ucl(alpha, data_df.shape[1], data_df.shape[0])
    qs = []
    for i in range(data_df.shape[0]):
        if(i < win_size or i >= end_idx):
            q = 0 #for data outside of window (to match behaviour CPD)
        else:
            cur_idx = win_size + i
            q = phase2controlpoint(data_df, alpha, cur_idx, win_size)
        qs.append(q)
    return qs, ucl

# parser = argparse.ArgumentParser()
# #parser.add_argument('--run_name', type=str, default="test_str")
# parser.add_argument('--data_fldr', type=str, default= "../MesonetData/InitTests")
# args = parser.parse_args()

data_fldr = "../MesonetData/InitTests"

for filename in os.listdir(data_fldr):
    filestr = os.path.join(data_fldr, filename)
    # checking if it is a file
    if os.path.isfile(filestr) and filename.__contains__("final"):
        
        str_end_idx = filename.find("final")
        NAME = filename[0: str_end_idx - 1]
        TRAIN_IDX = 7 * 288 #same as was run for CPD
        ALPHA = .05
        DATA_DF = pd.read_csv(data_fldr + "/" + NAME + "_final_data.csv") 
        STORM_DF = pd.read_csv(data_fldr + "/" + NAME + "_storm_data.csv")

        #####

        print("begin on " + NAME)
        end_idx = DATA_DF.shape[0] - TRAIN_IDX
        print(end_idx)
        qs1, ucl1 = sliding_control(DATA_DF, ALPHA,TRAIN_IDX, end_idx)

        qdf = pd.DataFrame(qs1)
        qdf['index'] = DATA_DF['YYYYMMDDhhmm']
        qdf = qdf.set_index('index')
        qdf.to_csv(NAME + "_q_ucl_ " + str(ucl1) + ".csv")

        qdf.clip(0, ucl1 + 50, inplace=True)
        qdf.plot()
        plt.axhline(ucl1, color='orange')
        
        #create datetime column for ease of comparison
        qdts = []
        for i in range(len(qdf.index.values)):
            qdts.append(datetime.fromisoformat(qdf.index[i]))
        qdf['datetimes'] = qdts

        mtimes = []
        #calculate location of metric occurrences
        for i in range(len(STORM_DF['BEGIN_DATE_TIME'])):
            #always round up the stormtime
            x = (qdf['datetimes'] - datetime.fromisoformat(STORM_DF.iloc[i]['BEGIN_DATE_TIME']))
            time = pos_argmin(x)
            clr=''
            stormtype = STORM_DF.iloc[i]['EVENT_TYPE']
            if(stormtype == 'Flood' or stormtype == 'Heavy Rain' or stormtype == 'Tropical Storm'):
                clr = 'blue'
            elif(stormtype == 'Winter Weather' or stormtype == 'Hail' or stormtype == 'Heavy Snow' or\
                 stormtype == 'Ice Storm' or stormtype == 'Blizzard' or 'Winter Storm'):
                 clr = 'maroon'
            elif(stormtype == 'Tornado' or stormtype == 'Thunderstorm Wind' or stormtype == 'Funnel Cloud' or\
                       stormtype == 'High Wind'):
                clr = 'grey'
            else:
                clr = 'black'    
            plt.axvline(time, color=clr, ls='--')
        plt.rcParams["figure.facecolor"] = 'white'
        plt.gcf().autofmt_xdate()
        plt.savefig("../Correlation-Changepoint-Detection/CPDplots/" + "HotellingTestsIndiv2" + "/" + NAME + ".png")