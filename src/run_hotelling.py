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

#phase 1 control chart, individaul
def phase1control(df, alpha):
    
    data = df.drop(['YYYYMMDDhhmm'], axis = 1)
    
    scaler = StandardScaler()
    df = scaler.fit_transform(data)
    p = data.shape[1]
    m = data.shape[0]
    
    means = []
    for i in range(len(data.columns)):
        v_i = df[i].mean()
        means.append(v_i)
    qs = []
    m = data.shape[0]
    s = np.linalg.inv(np.cov(np.transpose(data)))
    for i in tqdm(range(m)):
        v_i = data.iloc[i] - means
        v_i = v_i.to_list()
        vt_i = np.transpose(v_i)
        q = v_i @ s @ vt_i
        qs.append(q)
        
    ALPHA = alpha
    betappdf = beta.ppf(1 - (ALPHA /2), p/2, (m - p - 1) /2)
    mpart = (((m-1)**2) / m)
    ucl = mpart*betappdf
    
    return qs, ucl

def phase2control(df, alpha, train_idx):
  #  print(df.shape)
    data = df.drop(['YYYYMMDDhhmm'], axis = 1).to_numpy()

    train_data = data[0:train_idx]
    future_data = data[train_idx:]
    
    p = data.shape[1]
    m = data.shape[0]
    
    #means calculated from historical
    means = []
    for i in range(train_data.shape[1]):
        v_i = train_data[:,i].mean()
        means.append(v_i)
        
    qs = []
    m = data.shape[0]
    #s calculated from historical
    s = np.linalg.inv(np.cov(np.transpose(train_data)))
    
    for i in tqdm(range(data.shape[0])):
        v_i = data[i] - means
        #v_i = v_i
        vt_i = np.transpose(v_i)
        q = v_i @ s @ vt_i
        qs.append(q)
        
    ALPHA = alpha
    f_ucl = f.ppf(1 - (ALPHA /2), p, m - p)
    f_lcl = f.ppf((ALPHA /2), p, m - p)

    mpart = (p*(m+1)*(m-1)) / (m**2 - (m*p))
    ucl = mpart*f_ucl
    lcl = mpart*f_lcl
    return qs, lcl, ucl




def pos_argmin(times):
    argmin = -1
    amin = timedelta(days=370)
    for i in range(len(times)):
        if(times[i] > timedelta(days=0)):
            if(times[i] < amin):
                amin = times[i]
                argmin = i
    return argmin

parser = argparse.ArgumentParser()
#parser.add_argument('--run_name', type=str, default="test_str")
parser.add_argument('--data_fldr', type=str, default= "../MesonetData/InitTests")
args = parser.parse_args()

for filename in os.listdir(args.data_fldr):
    filestr = os.path.join(args.data_fldr, filename)
    # checking if it is a file
    if os.path.isfile(filestr) and filename.__contains__("final"):
        end_idx = filename.find("final")
        NAME = filename[0: end_idx - 1]

# ############ COULD EASILY BE ADAPTED TO FULLY PULL THESE FROM ARGS 
# ############ AND RUN IN THE SAME SCRIPT AS CPD
        #NAME = args.run_name
        TRAIN_IDX = 7 * 288 #same as was run for CPD
        ALPHA = .05
        DATA_DF = pd.read_csv(args.data_fldr + "/" + NAME + "_final_data.csv") 
        STORM_DF = pd.read_csv(args.data_fldr + "/" + NAME + "_storm_data.csv")

        #####

        print("begin on " + NAME)
        qs1, ucl1,lcl1 = phase2control(DATA_DF, ALPHA,TRAIN_IDX)

        qdf = pd.DataFrame(qs1)
        qdf['index'] = DATA_DF['YYYYMMDDhhmm']
        qdf = qdf.set_index('index')
        qdf.to_csv(NAME + "_q_ucl_ " + str(ucl1) + ".csv")

        qdf.clip(0, ucl1 + 50, inplace=True)
        qdf.plot()
        plt.axhline(ucl1, color='orange')
        plt.axhline(lcl1, color='orange')

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

        plt.axvline(TRAIN_IDX, color='red', ls='-')
        plt.gcf().autofmt_xdate()
        plt.savefig("../Correlation-Changepoint-Detection/CPDplots/" + "HotellingTestsIndiv" + "/" + NAME + ".png")
