import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import chi2, multivariate_normal, zscore
from tqdm import tqdm
from scipy.stats import norm
from numba import jit
import statistics
from math import sqrt
import argparse
sns.set()
#sns.set_style('dark')
sns.set_palette("viridis")
import matplotlib

from functools import reduce
import statistics

matplotlib.use('Agg')


def plot_confidence_interval(x, values, z=1.96, color='#2187bb', horizontal_line_width=0.25, constrain=False):
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    confidence_interval = z * stdev / sqrt(len(values))

    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    if constrain:
        if top <= 0.0:
            top = 0.0
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval
    plt.plot([x, x], [top, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)
    plt.plot(x, mean, 'o', color='#f44336')

    return mean, confidence_interval

def kesh_p_value(test_statistics):
    """
    Calculate z-score, then take survival function of normal dist
    """
    
    #z_scores = zscore(test_statistics)
    z_scores = (test_statistics - 0)/1 # substract mean and divide by standard deviation of true distribution?
    p_values = 1-norm.cdf(z_scores)
    
    return p_values

def gumbel_cdf(t):
    """
    Gumbel CDF for Xia/Cai
    """
    inner = -np.power(8*np.pi, -1/2)*np.exp(-t/2)
    
    return np.exp(inner)

def cai_p_value(test_statistics, dim):
    """
    Test stats difference with some logdim is gumbel
    
    Calculate with 1-cdf(.)
    """
    
    test_stats_adjusted = test_statistics - 4*np.log(dim) + np.log(np.log(dim))
    
    return 1-gumbel_cdf(test_stats_adjusted)



@jit(nopython=True)
def amoc_gen(alarms, first_detect_time, last_detect_time, max_time_detection, use_p_vals=False, thresholds=None):
    """
    Generate an AMOC curve.
    Arguments: 
        alarms: p-vals?
        first_detect_time
        max_time_detection: the maximum number of days to detect a fire (int)
    Returns: 
        FPR_array: false positive rates (list)
        detection_array: days to detection time (list)
    """
    #thresholds = np.arange(0, 1, .001)
    if not thresholds:
        thresholds = sorted(list(alarms.flatten()))
        #thresholds = sorted(list(set(alarms.flatten())))
    FPR_array = []
    detection_array = []

    # Check FPR and days to detection for each probability threshold
    for threshold in thresholds:
        detected = False
        detection_time = 0
        false_positives = 0
        true_negatives = 0
        
        for i in range(alarms.shape[0]):
        #for p_value in alarms: # not p-value right now
            
            # Two cases: false positive and true positive
            if use_p_vals:
                t_value = alarms[i, :].min()
            else:
                t_value = alarms[i, :].max()
            
            if use_p_vals:
                if t_value <= threshold:
                    if detection_time < first_detect_time or detection_time > last_detect_time: # Check for false positive
                        false_positives += 1

                    else: # True positive: fire detected
                        detected = True
                        detection_array.append(detection_time - first_detect_time)
                        break

                else: # Two cases: false negative and true negative
                    if detection_time < first_detect_time: # Check for true negative -- needed for FPR
                        true_negatives += 1
                detection_time += 1
            else:
                if t_value >= threshold:
                    if detection_time < first_detect_time or detection_time > last_detect_time: # Check for false positive
                        false_positives += 1

                    else: # True positive: fire detected
                        detected = True
                        detection_array.append(detection_time - first_detect_time)
                        break

                else: # Two cases: false negative and true negative
                    if detection_time < first_detect_time: # Check for true negative -- needed for FPR
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

def apply_fdr_correction(p_vals_all, alpha=0.05):
    corrected_p_vals_all = []
    for i in range(p_vals_all.shape[0]):
        rejected, corrected_p_vals_i = fdrcorrection(p_vals_all[i], alpha=alpha)
        corrected_p_vals_all.append(corrected_p_vals_i)
    return np.array(corrected_p_vals_all)


def average_AMOC_new(fprs, dts, debug_title="Ours"):
    thresholds = np.arange(0.0, 1.0, 0.001) # list of possible FPR values
    mean_days_per_threshold = []
    fpr_dict = {}
    for thresh in thresholds:
        detect_times_per_thresh = []
        # iterate over each curve
        for i in range(len(fprs)):
            #min_dt_per_fpr = np.interp(thresh, fprs[i], dts[i])
            curr_fpr = fprs[i]
            curr_dt = dts[i]
            fpr_diff = np.abs(curr_fpr-thresh)
            min_idxs = np.where(fpr_diff == fpr_diff.min())[0]
            min_dt_per_fpr = curr_dt[min_idxs].min()
            detect_times_per_thresh.append(min_dt_per_fpr)
        mean_val = sum(detect_times_per_thresh)/len(fprs)
        mean_days_per_threshold.append(mean_val)
        if thresh == 0.01 or thresh == 0.05 or thresh == 0.1:
            fpr_dict[thresh] = detect_times_per_thresh
            #print("FPR {} Mean {} Min {} Max {}".format(thresh, statistics.mean(detect_times_per_thresh), min(detect_times_per_thresh), max(detect_times_per_thresh)))
            sns.histplot(detect_times_per_thresh, kde=True)
            plt.title("Detect Times FPR: {}".format(thresh))
            plt.savefig("debugging_figs/{}_{}_fpr_fig.png".format(debug_title, thresh))
            plt.close()
    
    return np.array(mean_days_per_threshold), thresholds, fpr_dict

def average_AMOC(AMOC_points):
    """
    Generate an average AMOC curve.
    Arguments: 
        alarms_dataset: List of lists of tuples, where points are FPRs and associated
                        detection times
                        [ [(FPR, detection time), (FPR, detection time)], ...] 
                          [(FPR, detection time), (FPR, detection time)], ...] 
                          ...
                        ]
    Returns: 
        mean_days_per_threshold: list of mean days to detection for each threshold
    """
    mean_days_per_threshold = []
    thresholds = np.arange(0.0, 1.0, 0.001) # list of possible FPR values
    # Find mean detection time for each threshold
    fpr_dict = {}
    for k in tqdm(range(len(thresholds))):
        threshold = thresholds[k]
        summed_days = 0
        detect_times_per_thresh = []

        # Approximate detection time for threshold using closest FPR in each simulation
        for i in range(0, len(AMOC_points)):
            current_dataset = AMOC_points[i]  # Get list of alarm values
            closest_pair = current_dataset[0] # Find pair with closest probability threshold
            
            for j in range(0, len(current_dataset)):
                curr_point = current_dataset[j]     # point: (FPR, days to detection)
                if np.abs(curr_point[0] - threshold) < np.abs(closest_pair[0] - threshold):
                    closest_pair = curr_point
            
            # sum the detection times of each dataset that correspond with 
            # probability threshold
            summed_days += closest_pair[1]
            detect_times_per_thresh.append(closest_pair[1])
        
        mean_days = summed_days / len(AMOC_points)
        mean_days_per_threshold.append(mean_days)

        if threshold == 0.01 or threshold==0.05 or threshold==0.1:
            fpr_dict[threshold] = detect_times_per_thresh
    
    return np.array(mean_days_per_threshold), thresholds, fpr_dict
    
def symmetrize_from_vector(a, dim):
    """
    Turns a vector of lower triangular matrix entries into symmetric matrix
    """
    A = np.zeros((dim,dim))
    A[np.tril_indices(A.shape[0], k = 0)] = a
    A = A + A.T - np.diag(np.diag(A))

    return A

def get_h_matrices(full_path, num_mats, dim):
    H_s = []
    for i in range(num_mats):
        H_i = np.loadtxt(os.path.join(full_path, 'matrix_{}.csv'.format(i)))
        H_s.append(symmetrize_from_vector(H_i, dim))
    return np.array(H_s)


# TONY CAI

@jit(nopython=True)
def indicator_global_stat(p, alpha=0.01):
    q_alpha = -np.log(8*np.pi)-2*np.log(np.log(np.power(1-alpha, -1)))
    threshold = q_alpha+4*np.log(p)-np.log(np.log(p))

    return threshold

def amoc_gen_vectorized(lrt_values, first_detect_time, last_detect_time, max_time_detection, thresholds=None, p_values=False):
    if not thresholds:
        thresholds = sorted(list(lrt_values))

    fpr_array = []
    detect_array = []

    for thresh in thresholds:
        if p_values: # if we're using p-values, lower is the cutoff
            exceed_mask = (lrt_values <= thresh) 
        else: # if we're using LRT-values, higher is the cutoff
            exceed_mask = (lrt_values >= thresh)

        # calculate false_positives
        early_fp = np.sum(exceed_mask[0:first_detect_time])
        late_fp = np.sum(exceed_mask[last_detect_time+1:])

        # calculate true_negatives
        early_tn = np.sum(~exceed_mask[0:first_detect_time])
        late_tn = np.sum(~exceed_mask[last_detect_time+1:])

        cp_slice = lrt_values[first_detect_time:last_detect_time]
        cp_mask_slice = exceed_mask[first_detect_time:last_detect_time]
        # we didn't find it
        if np.sum(cp_mask_slice) < 1:
            detect_array.append(max_time_detection)
        # we found it, take the earliest True
        else:
            earliest_time = np.where(cp_mask_slice==True)[0][0]
            #print(earliest_time)
            detect_array.append(earliest_time) # earliest detection
        
        false_positives = early_fp + late_fp
        true_negatives = early_tn + late_tn
        if false_positives + true_negatives < 1: # shouldn't happen, but if everything is storm
            fp_rate = 0.0
        else:
            fp_rate = false_positives/(false_positives+true_negatives)
        fpr_array.append(fp_rate)
    
    return fpr_array, detect_array



@jit(nopython=True)
def amoc_gen_cai(alarms, first_detect_time, last_detect_time, max_time_detection, p, thresholds=None):
    """
    Generate an AMOC curve.
    Arguments: 
        alarms: test_stat-vals
        first_detect_time
        max_time_detection: the maximum number of days to detect a fire (int)
    Returns: 
        FPR_array: false positive rates (list)
        detection_array: days to detection time (list)
    """
    #thresholds = np.arange(0, 1, .001)
    if not thresholds:
        thresholds = sorted(list(alarms))
        #print(thresholds)
    #thresholds = sorted(list(np.linspace(alarms.min(), alarms.max(), 6349)))
    FPR_array = []
    detection_array = []
    prev_detection_time = np.inf
    prev_fp_rate = np.inf
    # Check FPR and days to detection for each probability threshold
    for threshold in thresholds:
        #print(threshold)
        #thr = indicator_global_stat(p, alpha=threshold)
        thr = threshold # use the set of possible values
        detected = False
        detection_time = 0
        curr_detect_time = 0
        false_positives = 0
        true_negatives = 0
        for lrt_value in alarms:
            # Two cases: false positive and true positive
            if lrt_value >= thr:
                if detection_time < first_detect_time or detection_time > last_detect_time: # Check for false positive
                    false_positives += 1
                
                else: # True positive: fire detected
                    detected = True
                    detection_array.append(detection_time - first_detect_time)
                    curr_detect_time = detection_time - first_detect_time
                    # if curr_detect_time < prev_detection_time:
                    #     print("Threshold", thr, "Curr Detect Time", curr_detect_time, "Prev Detect Time", prev_detection_time)
                    prev_detection_time = curr_detect_time
                    break
            
            else: # Two cases: false negative and true negative
                if detection_time < first_detect_time or detection_time > last_detect_time: # Check for true negative -- needed for FPR
                    true_negatives += 1
            detection_time += 1

        if false_positives + true_negatives != 0:
            fp_rate = (false_positives) / (false_positives + true_negatives)
            
        else:
            fp_rate = 0
        # curr_fp_rate = fp_rate
        FPR_array.append(fp_rate)
        # if curr_fp_rate > prev_fp_rate:
        #     print(curr_fp_rate)
        # prev_fp_rate = curr_fp_rate
        if not detected:
            curr_detect_time = max_time_detection
            # if curr_detect_time < prev_detection_time:
            #     print("Threshold", thr, "Curr Detect Time", curr_detect_time, "Prev Detect Time", prev_detection_time)
            prev_detection_time = curr_detect_time
            detection_array.append(max_time_detection)

    return FPR_array, detection_array

def amoc_lrt_vals(lrt_vals, first_possible_detect_time, last_possible_detect_time, thresholds=None, p_values=False):
    fprs = []
    detections = []
    
    # fprs, detections = amoc_gen_cai(lrt_vals, 
    #                                 first_detect_time=first_possible_detect_time, 
    #                                 last_detect_time=last_possible_detect_time, 
    #                                 max_time_detection=last_possible_detect_time-first_possible_detect_time,
    #                                 p=dim)
    fprs, detections = amoc_gen_vectorized(lrt_vals, 
                                    first_detect_time=first_possible_detect_time, 
                                    last_detect_time=last_possible_detect_time, 
                                    max_time_detection=last_possible_detect_time-first_possible_detect_time,
                                    p_values=p_values
                                    )

    fprs = np.array(fprs)
    detections = np.array(detections)
    
    return fprs, detections

#@jit(nopython=True)
def amoc_gen_p_vals(alarms, first_detect_time, last_detect_time, max_time_detection, p, thresholds=None):
    """
    Generate an AMOC curve.
    Arguments: 
        alarms: p-vals?
        first_detect_time
        max_time_detection: the maximum number of days to detect a fire (int)
    Returns: 
        FPR_array: false positive rates (list)
        detection_array: days to detection time (list)
    """
    #thresholds = np.arange(0, 1, .001)
    if thresholds is None:
        thresholds = sorted(list(alarms))
        #print(thresholds)
    #thresholds = sorted(list(np.linspace(alarms.min(), alarms.max(), 6349)))
    FPR_array = []
    detection_array = []

    # Check FPR and days to detection for each probability threshold
    for threshold in thresholds:
        #print(threshold)
        #thr = indicator_global_stat(p, alpha=threshold)
        thr = threshold # use the set of possible values
        detected = False
        detection_time = 0
        false_positives = 0
        true_negatives = 0
        for p_value in alarms:
            # Two cases: false positive and true positive
            if p_value <= thr:
                if detection_time < first_detect_time or detection_time > last_detect_time: # Check for false positive
                    false_positives += 1
                
                else: # True positive: fire detected
                    detected = True
                    detection_array.append(detection_time - first_detect_time)
                    break
            
            else: # Two cases: false negative and true negative
                if detection_time < first_detect_time or detection_time > last_detect_time: # Check for true negative -- needed for FPR
                    true_negatives += 1
            detection_time += 1

        if false_positives + true_negatives != 0:
            fp_rate = (false_positives) / (false_positives + true_negatives)
            
        else:
            fp_rate = 0

        FPR_array.append(fp_rate)
        if not detected:
            detection_array.append(max_time_detection)
    
    return FPR_array, detection_array

def amoc_p_vals(p_vals, dim, first_possible_detect_time, last_possible_detect_time, thresholds=None):
    fprs = []
    detections = []
    
    fprs, detections = amoc_gen_p_vals(p_vals, 
                                       first_detect_time=first_possible_detect_time, 
                                       last_detect_time=last_possible_detect_time, 
                                       max_time_detection=last_possible_detect_time-first_possible_detect_time,
                                       p=dim,
                                       thresholds=thresholds)

    fprs = np.array(fprs)
    detections = np.array(detections)
    
    return fprs, detections


def main_sims():
    curr_path = os.getcwd()
    save_path = os.path.join(curr_path, 'amoc_figs/')
    seeds = np.arange(50, 70)
    #seeds = np.arange(50, 54)
    #sim_types = ['anderson_residual_block', 'anderson_residual_unstructured', 'orthogonal']
    sim_types = ['anderson_residual_unstructured']
    #sim_types = ['cai_model_one']
    dims = [20, 40, 60, 80]
    #dims = [20,40,60]
    #dims = [80]
    #dims = [24, 30, 36, 42, 48, 56, 64]
    #dims = [20, 24, 30]
    #dims=[36]

    our_results = {}
    cai_results = {}
    kesh_results = {}
    for curr_dim in dims:
        curr_dim_path = os.path.join(save_path, str(curr_dim))
        if not os.path.exists(curr_dim_path):
            os.mkdir(curr_dim_path)
        for sim_type in sim_types:
            print(sim_type)
            our_seed_dict = {}
            cai_seed_dict = {}
            kesh_seed_dict = {}
            fpr_detect_pairs_ours = []
            fpr_detect_pairs_cai = []
            fpr_detect_pairs_kesh = []
            lol_ours_fprs = []
            lol_ours_dts = []
            lol_cai_fprs = []
            lol_cai_dts = []
            lol_kesh_fprs = []
            lol_kesh_dts = []
            statset_ours = []
            statset_cai = []
            statset_kesh = []
            for curr_seed in seeds:
                cpd_path='./results/simulation_results/{}_{}/{}/lrt_vals.csv'.format(sim_type, curr_dim, curr_seed)
                xia_path='./results/simulation_results_cai/{}_{}/{}/global_test_vals.csv'.format(sim_type, curr_dim, curr_seed)
                kesh_path='./results/simulation_results_kesh/{}_{}/{}/global_test_vals.csv'.format(sim_type, curr_dim, curr_seed)
                cpd_vals = np.loadtxt(cpd_path, delimiter=',')
                cutoff = cpd_vals.shape[1]//2
                pvals_cpd = cpd_vals[:, cutoff:][100:] # just the p-values
                stats_cpd = cpd_vals[:, 0:cutoff][100:] # just the stats
                xia_cpd = np.loadtxt(xia_path, delimiter=',')
                kesh_cpd = np.loadtxt(kesh_path, delimiter=',')
                #pvals_cpd_padded = np.pad(pvals_cpd, [(100, 100), (0,0)], mode='constant')
                #stats_cpd_padded = np.pad(stats_cpd, [(100, 100), (0,0)], mode='constant')[0:-100, :]
                #stats_cpd = stats_cpd_padded
                #xia_cpd_padded = np.pad(xia_cpd, 100, mode='constant')
                #cp_location = stats_cpd_padded.shape[0]//2
                cp_location = 300 # THIS IS THE ACTUAL LOCATION, FIRST DETECTION TIME IS (ACTUAL LOCATION - WINDOW SIZE + 1)
                window_size = 100
                first_d_time = cp_location-window_size+1
                last_d_time = 600
    #             lrt_result_cpd, detect_result_cpd = amoc_lrt_vals(stats_cpd, 
    #                                                               first_d_time, 
    #                                                               last_d_time,
    #                                                               use_p_vals=False,
    #                                                               thresholds=None
    #                                                              )
    #             lrt_result_cpd, detect_result_cpd = amoc_lrt_vals_cai(stats_cpd.max(axis=1), 
    #                                                                   first_possible_detect_time=first_d_time,
    #                                                                   last_possible_detect_time=last_d_time,
    #                                                                   dim=curr_dim
    #                                                                  )
    #             lrt_result_cai, detect_result_cai = amoc_lrt_vals_cai(xia_cpd, 
    #                                                                   first_possible_detect_time=first_d_time,
    #                                                                   last_possible_detect_time=last_d_time,
    #                                                                   dim=curr_dim
    #                                                                  )
    #             lrt_result_kesh, detect_result_kesh = amoc_lrt_vals_cai(kesh_cpd, 
    #                                                                   first_possible_detect_time=first_d_time,
    #                                                                   last_possible_detect_time=last_d_time,
    #                                                                   dim=curr_dim
    #                                                                  )
                pvals_cpd = pvals_cpd.min(axis=1)
                pvals_cai = cai_p_value(xia_cpd, curr_dim)
                pvals_kesh = kesh_p_value(kesh_cpd)
                
                #all_thresholds = sorted(list(set(np.concatenate((pvals_cpd, pvals_cai, pvals_kesh)))))
                all_thresholds = None
                
                lrt_result_cpd, detect_result_cpd = amoc_lrt_vals(pvals_cpd, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=all_thresholds, p_values=True)
                lrt_result_cai, detect_result_cai = amoc_lrt_vals(pvals_cai, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=all_thresholds, p_values=True)
                #lrt_result_kesh, detect_result_kesh = amoc_lrt_vals(pvals_kesh, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=all_thresholds, p_values=True)

                #lrt_result_cpd, detect_result_cpd = amoc_lrt_vals(stats_cpd.max(axis=1), first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=None, p_values=False)
                #lrt_result_cai, detect_result_cai = amoc_lrt_vals(xia_cpd, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=None, p_values=False)
                lrt_result_kesh, detect_result_kesh = amoc_lrt_vals(kesh_cpd, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=None, p_values=False)

                # lrt_result_cpd, detect_result_cpd = amoc_p_vals(pvals_cpd, 
                #                                                 first_possible_detect_time=first_d_time,
                #                                                 last_possible_detect_time=last_d_time,
                #                                                 dim=curr_dim,
                #                                                 thresholds=all_thresholds
                #                                                 )
                # lrt_result_cai, detect_result_cai = amoc_p_vals(pvals_cai, 
                #                                                 first_possible_detect_time=first_d_time,
                #                                                 last_possible_detect_time=last_d_time,
                #                                                 dim=curr_dim,
                #                                                 thresholds=all_thresholds
                #                                                 )
                # lrt_result_kesh, detect_result_kesh = amoc_p_vals(pvals_kesh, 
                #                                                 first_possible_detect_time=first_d_time,
                #                                                 last_possible_detect_time=last_d_time,
                #                                                 dim=curr_dim,
                #                                                 thresholds=all_thresholds
                #                                                 )
                
                our_seed_dict[curr_seed] = (lrt_result_cpd, detect_result_cpd)
                cai_seed_dict[curr_seed] = (lrt_result_cai, detect_result_cai)
                kesh_seed_dict[curr_seed] = (lrt_result_kesh, detect_result_kesh)
                our_results[sim_type] = our_seed_dict
                cai_results[sim_type] = cai_seed_dict
                kesh_results[sim_type] = kesh_seed_dict
                print(pvals_cpd.shape, stats_cpd.shape, stats_cpd.max(axis=1).shape, xia_cpd.shape, kesh_cpd.shape)

                sorted_lrt_result_cpd = lrt_result_cpd #sorted(lrt_result_cpd, reverse=True)
                sorted_lrt_result_cai = lrt_result_cai #sorted(lrt_result_cai, reverse=True)
                sorted_lrt_result_kesh = lrt_result_kesh #sorted(lrt_result_kesh, reverse=True)
                sorted_detect_result_cpd = detect_result_cpd #sorted(detect_result_cpd, reverse=False)
                sorted_detect_result_cai = detect_result_cai #sorted(detect_result_cai, reverse=False)
                sorted_detect_result_kesh = detect_result_kesh #sorted(detect_result_kesh, reverse=False)

                plt.step(sorted_lrt_result_cpd, sorted_detect_result_cpd, '-.r', 
                        label='Cai', where='post')
                plt.step(sorted_lrt_result_kesh, sorted_detect_result_kesh, '-.g', 
                        label='Kesh', where='post')
                plt.step(sorted_lrt_result_cai, sorted_detect_result_cai, '-.b', 
                        label='Ours', where='post')
                plt.ylabel("Detect Time")
                plt.xlabel("FPR")
                plt.title("{} Seed {}".format(sim_type, curr_seed))
                plt.legend(loc='best')
                plt.savefig(os.path.join(curr_dim_path, 'amoc_{}_seed_{}_dim_{}.png'.format(sim_type, curr_seed, curr_dim)))
                plt.close()


                merged_list_ours = [(sorted_lrt_result_cpd[i], 
                                    sorted_detect_result_cpd[i]) for i in range(0, len(lrt_result_cpd))]
                merged_list_cai = [(sorted_lrt_result_cai[i], 
                                    sorted_detect_result_cai[i]) for i in range(0, len(lrt_result_cai))]
                merged_list_kesh = [(sorted_lrt_result_kesh[i], 
                                    sorted_detect_result_kesh[i]) for i in range(0, len(lrt_result_kesh))]
                fpr_detect_pairs_ours.append(merged_list_ours)
                fpr_detect_pairs_cai.append(merged_list_cai)
                fpr_detect_pairs_kesh.append(merged_list_kesh)

                # list-of-list appending
                lol_ours_fprs.append(sorted_lrt_result_cpd)
                lol_ours_dts.append(sorted_detect_result_cpd)
                lol_cai_fprs.append(sorted_lrt_result_cai)
                lol_cai_dts.append(sorted_detect_result_cai)
                lol_kesh_fprs.append(sorted_lrt_result_kesh)
                lol_kesh_dts.append(sorted_detect_result_kesh)
                
                statset_ours.append(stats_cpd.flatten())
                statset_cai.append(xia_cpd.flatten())
                statset_kesh.append(kesh_cpd.flatten())
            
            #means_per_threshold_ours, thresholds_ours, fpr_dict_ours = average_AMOC(fpr_detect_pairs_ours)
            #means_per_threshold_cai, thresholds_cai, fpr_dict_cai = average_AMOC(fpr_detect_pairs_cai)
            #means_per_threshold_kesh, thresholds_kesh, fpr_dict_kesh = average_AMOC(fpr_detect_pairs_kesh)
            means_per_threshold_ours, thresholds_ours, fpr_dict_ours = average_AMOC_new(fprs=lol_ours_fprs, dts=lol_ours_dts)
            means_per_threshold_cai, thresholds_cai, fpr_dict_cai = average_AMOC_new(fprs=lol_cai_fprs, dts=lol_cai_dts)
            means_per_threshold_kesh, thresholds_kesh, fpr_dict_kesh = average_AMOC_new(fprs=lol_kesh_fprs, dts=lol_kesh_dts)

            plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9], ["0.01_ours", "0.01_cai", "0.01_kesh", "0.05_ours", "0.05_cai", "0.05_kesh", "0.1_ours", "0.1_cai", "0.1_kesh"])
            plt.xticks(rotation=45, ha='right')
            plot_idx_counter = 1
            ylim_max = 0.0
            for fpr_r in [0.01, 0.05, 0.1]:
                our_d_vals = fpr_dict_ours[fpr_r]
                cai_d_vals = fpr_dict_cai[fpr_r]
                kesh_d_vals = fpr_dict_kesh[fpr_r]
                our_max = max(our_d_vals)
                cai_max = max(cai_d_vals)
                kesh_max = max(kesh_d_vals)
                glob_max = max([our_max, cai_max, kesh_max])
                if glob_max > ylim_max:
                    ylim_max = glob_max
                plot_confidence_interval(x=plot_idx_counter, values=our_d_vals, z=1.96, color='#2187bb', horizontal_line_width=0.25)
                plot_idx_counter += 1
                plot_confidence_interval(x=plot_idx_counter, values=cai_d_vals, z=1.96, color='#2187bb', horizontal_line_width=0.25)
                plot_idx_counter += 1
                plot_confidence_interval(x=plot_idx_counter, values=kesh_d_vals, z=1.96, color='#2187bb', horizontal_line_width=0.25)
                plot_idx_counter += 1
            plt.ylim(-10.0, 40.0)
            plt.title("Confidence Intervals {} Dim {}".format(sim_type, curr_dim))
            plt.ylabel("Detection Time")
            plt.xlabel("FPR")
            plt.tight_layout()
            plt.savefig(os.path.join(curr_dim_path, 'conf_avg_amoc_{}_dim_{}.png'.format(sim_type, curr_dim)))
            plt.close()
            
            """
            ###########################
            Filter out highest detection time - unrealistic
            and high FPR - also unrealistic
            """
            threshold_idx_mask = np.where(thresholds_ours <= 0.2) # THESE ARE IDENTICAL INDICES ACROSS THE ALGORITHMS
            means_per_threshold_ours = means_per_threshold_ours[threshold_idx_mask]
            means_per_threshold_cai = means_per_threshold_cai[threshold_idx_mask]
            means_per_threshold_kesh = means_per_threshold_kesh[threshold_idx_mask]
            thresholds_ours = thresholds_ours[threshold_idx_mask]
            thresholds_cai = thresholds_cai[threshold_idx_mask]
            thresholds_kesh = thresholds_kesh[threshold_idx_mask]

            detect_idx_mask_ours = np.where(means_per_threshold_ours <= 200) # maximum detection time, 2*window_size
            detect_idx_mask_cai = np.where(means_per_threshold_cai <= 200)
            detect_idx_mask_kesh = np.where(means_per_threshold_kesh <= 200)
            detect_idx_mask = reduce(np.intersect1d, (detect_idx_mask_ours, detect_idx_mask_cai, detect_idx_mask_kesh)) # union of all idxs
            
            # means_per_threshold_ours = means_per_threshold_ours[detect_idx_mask] # 4 seems fine
            # means_per_threshold_cai = means_per_threshold_cai[detect_idx_mask]
            # means_per_threshold_kesh = means_per_threshold_kesh[detect_idx_mask]
            # thresholds_ours = thresholds_ours[detect_idx_mask]
            # thresholds_cai = thresholds_cai[detect_idx_mask]
            # thresholds_kesh = thresholds_kesh[detect_idx_mask]


            """
            ###########################
            """
            plt.plot(thresholds_ours, means_per_threshold_ours, '.b-', label='Ours')
            plt.plot(thresholds_cai, means_per_threshold_cai, '.r-', label='Cai')
            plt.plot(thresholds_kesh, means_per_threshold_kesh, '.g-', label='Kesh')
            plt.xlabel('FPR')
            plt.ylabel('Detection Time')
            plt.title('Average AMOC Curve for {} Dim {}'.format(sim_type, curr_dim))
            plt.ylim(0.0, 20.0)
            plt.legend(loc='best')
            plt.savefig(os.path.join(curr_dim_path, 'avg_amoc_{}_dim_{}.png'.format(sim_type, curr_dim)))
            plt.close()

def main_mesonet(storm_name='center'):
    """
    In progress for MesoNet data collection

    Will be more involved. Mainly for CP location identification with storm pairings
    """
    window_size = 400 # this is fixed currently
    dim = 35 # this is fixed currently
    curr_path = os.getcwd()
    save_path = os.path.join(curr_path, 'amoc_figs/')
    curr_storm_save_path = save_path+"mesonet/"+storm_name
    cpd_path = 'results/{}_storm'.format(storm_name)
    kesh_path = 'results_kesh/mesonet'
    cai_path = 'results_cai/mesonet'
    storm_data_path = '../data/out_{}'.format(storm_name)
    dir_files = os.listdir(cpd_path)
    storm_dir_files = os.listdir(storm_data_path)
    storm_nums = []
    for fi in dir_files:
        if 'center_storm' in fi:
            storm_nums.append(int(fi.split('_')[-1].split('.')[0]))

    storm_files = []
    for num in storm_nums:
        storm_files.append('{}_storm_{}_storm_data.csv'.format(storm_name, num))
    
    our_results = {}
    kesh_results = {}
    fpr_detect_pairs_ours = []
    fpr_detect_pairs_cai = []
    fpr_detect_pairs_kesh = []
    lol_ours_fprs = []
    lol_ours_dts = []
    lol_cai_fprs = []
    lol_cai_dts = []
    lol_kesh_fprs = []
    lol_kesh_dts = []
    unique_tuples = []
    for num in sorted(storm_nums):
        cpd_vals = np.loadtxt(os.path.join(cpd_path, "{}_storm_{}.csv".format(storm_name, num)), delimiter=',')
        cutoff = cpd_vals.shape[1]//2 # split test stats and p-vals
        pvals_cpd = cpd_vals[:, cutoff:][window_size:] # just the p-values, exclude first window size
        stats_cpd = cpd_vals[:, 0:cutoff][window_size:] # just the stats, exclude first window size
        kesh_vals = np.loadtxt(os.path.join(kesh_path, "{}_storm_{}_final_data.csv_global_test_vals.csv".format(storm_name, num)), delimiter=',')
        cai_vals = np.loadtxt(os.path.join(cai_path, "{}_storm_{}_final_data.csv_global_test_vals.csv".format(storm_name, num)), delimiter=',')
        actual_final_data = pd.read_csv(os.path.join(storm_data_path, '{}_storm_{}_final_data.csv').format(storm_name, num))
        actual_final_data['YYYYMMDDhhmm'] = pd.to_datetime(actual_final_data['YYYYMMDDhhmm'])
        actual_storm_data = pd.read_csv(os.path.join(storm_data_path, '{}_storm_{}_storm_data.csv').format(storm_name, num))
        storm_start = pd.to_datetime(actual_storm_data.iloc[2]).dt.round("5min") # always index 2 for the start time
        storm_end = pd.to_datetime(actual_storm_data.iloc[3]).dt.round("5min") # always index 3 for the end time
        storm_type = actual_storm_data.iloc[1].item()
        storm_start_idx = actual_final_data[actual_final_data['YYYYMMDDhhmm']==storm_start.item()].index.item()
        storm_end_idx = actual_final_data[actual_final_data['YYYYMMDDhhmm']==storm_end.item()].index.item()
        if storm_start.item() == storm_end.item():
            continue
        else:
            curr_tuple = (storm_type, storm_start.item(), storm_end.item())
            if curr_tuple in list(set(unique_tuples)):
                continue
            else:
                unique_tuples.append(curr_tuple)
        cp_location = storm_start_idx
        # subtract off window_size to handle the sliding window edges at start/end of data
        first_d_time = cp_location - window_size + 1 - window_size
        last_d_time = cp_location + storm_end_idx - storm_start_idx + 1 - window_size# how long is the storm contained in the window as it slides over the data - at least window_size times (right to left)
        print("Storm type {}; Duration {}".format(storm_type, storm_end_idx-storm_start_idx))
        #print(pvals_cpd.shape, stats_cpd.shape, kesh_vals.shape, actual_final_data.shape)
        
        
        #print(stats_cpd.min(), stats_cpd.max(), pvals_cpd.max())
        pvals_cpd = pvals_cpd.min(axis=1)
        pvals_kesh = kesh_p_value(kesh_vals)
        pvals_cai = cai_p_value(cai_vals, dim)

        """
        Cut out points after storm
        """
        pvals_cpd = pvals_cpd[0:last_d_time+1]
        pvals_kesh = pvals_kesh[0:last_d_time+1]
        pvals_cai = pvals_cai[0:last_d_time+1]
        stats_cpd = stats_cpd.max(axis=1)[0:last_d_time+1]
        kesh_vals = kesh_vals[0:last_d_time+1]
        cai_vals = cai_vals[0:last_d_time+1]
        """
        """

        all_thresholds = sorted(list(set(np.concatenate((pvals_cpd, pvals_kesh)))))

        # lrt_result_cpd, detect_result_cpd = amoc_p_vals(pvals_cpd, 
        #                                                 first_possible_detect_time=first_d_time,
        #                                                 last_possible_detect_time=last_d_time,
        #                                                 dim=None, # deprecated
        #                                                 thresholds=all_thresholds
        #                                                 )
        
        lrt_result_cpd, detect_result_cpd = amoc_lrt_vals(stats_cpd, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=None, p_values=False)
        lrt_result_kesh, detect_result_kesh = amoc_lrt_vals(kesh_vals, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=None, p_values=False)
        lrt_result_cai, detect_result_cai = amoc_lrt_vals(cai_vals, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=None, p_values=False)
        print(detect_result_cpd.min(), detect_result_cpd.max(), detect_result_cpd.mean())
        # lrt_result_cpd, detect_result_cpd = amoc_lrt_vals(pvals_cpd, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=None, p_values=True)
        # lrt_result_kesh, detect_result_kesh = amoc_lrt_vals(pvals_kesh, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=None, p_values=True)
        # lrt_result_cai, detect_result_cai = amoc_lrt_vals(pvals_cai, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=None, p_values=True)
        
        #print(pvals_cpd.shape, stats_cpd.shape, kesh_vals.shape, cai_vals.shape)
        # plt.plot(lrt_result_cpd)
        # plt.savefig('./fpr_temp_{}.png'.format(num))
        # plt.close()
        # plt.plot(detect_result_cpd)
        # plt.savefig('./detect_temp_{}.png'.format(num))
        # plt.close()
        sorted_lrt_result_cpd = lrt_result_cpd  # sorted(lrt_result_cpd, reverse=True) # 
        sorted_lrt_result_cai = lrt_result_cai #sorted(lrt_result_cai, reverse=True)
        sorted_lrt_result_kesh = lrt_result_kesh  #sorted(lrt_result_kesh, reverse=True) #
        sorted_detect_result_cpd = detect_result_cpd  #sorted(detect_result_cpd, reverse=False) # 
        sorted_detect_result_cai = detect_result_cai #sorted(detect_result_cai, reverse=False)
        sorted_detect_result_kesh = detect_result_kesh  # sorted(detect_result_kesh, reverse=False) #

        merged_list_ours = [(sorted_lrt_result_cpd[i], 
                            sorted_detect_result_cpd[i]) for i in range(0, len(lrt_result_cpd))]
        merged_list_cai = [(sorted_lrt_result_cai[i], 
                            sorted_detect_result_cai[i]) for i in range(0, len(lrt_result_cai))]
        merged_list_kesh = [(sorted_lrt_result_kesh[i], 
                            sorted_detect_result_kesh[i]) for i in range(0, len(lrt_result_kesh))]
        fpr_detect_pairs_ours.append(merged_list_ours)
        fpr_detect_pairs_cai.append(merged_list_cai)
        fpr_detect_pairs_kesh.append(merged_list_kesh)
        lol_ours_fprs.append(sorted_lrt_result_cpd)
        lol_ours_dts.append(sorted_detect_result_cpd)
        lol_cai_fprs.append(sorted_lrt_result_cai)
        lol_cai_dts.append(sorted_detect_result_cai)
        lol_kesh_fprs.append(sorted_lrt_result_kesh)
        lol_kesh_dts.append(sorted_detect_result_kesh)

        plt.step(sorted_lrt_result_cpd, sorted_detect_result_cpd, '-.r', 
                        label='Cai', where='post')
        plt.step(sorted_lrt_result_kesh, sorted_detect_result_kesh, '-.g', 
                label='Kesh', where='post')
        plt.step(sorted_lrt_result_cai, sorted_detect_result_cai, '-.b', 
                label='Ours', where='post')
        plt.ylabel("Detect Time (5min Intervals)")
        plt.xlabel("FPR")
        plt.title("{} Storm_{}".format(storm_name, num))
        plt.legend(loc='best')
        plt.savefig(os.path.join(curr_storm_save_path, 'amoc_{}_{}.png'.format(storm_name, num)))
        plt.close()
    #means_per_threshold_ours, thresholds_ours, fpr_dict_ours = average_AMOC(fpr_detect_pairs_ours)
    #means_per_threshold_cai, thresholds_cai, fpr_dict_cai = average_AMOC(fpr_detect_pairs_cai)
    #means_per_threshold_kesh, thresholds_kesh, fpr_dict_kesh = average_AMOC(fpr_detect_pairs_kesh)
    means_per_threshold_ours, thresholds_ours, fpr_dict_ours = average_AMOC_new(fprs=lol_ours_fprs, dts=lol_ours_dts, debug_title='Ours')
    means_per_threshold_cai, thresholds_cai, fpr_dict_cai = average_AMOC_new(fprs=lol_cai_fprs, dts=lol_cai_dts, debug_title='Cai')
    means_per_threshold_kesh, thresholds_kesh, fpr_dict_kesh = average_AMOC_new(fprs=lol_kesh_fprs, dts=lol_kesh_dts, debug_title='Kesh')

    
    plot_idx_counter = 1
    ylim_max = 0.0
    for fpr_r in [0.01, 0.05, 0.1]:
        plt.xticks([1, 2, 3], ["{}_ours".format(fpr_r), "{}_cai".format(fpr_r), "{}_kesh".format(fpr_r)])
        plt.xticks(rotation=45, ha='right')
        our_d_vals = fpr_dict_ours[fpr_r]
        cai_d_vals = fpr_dict_cai[fpr_r]
        kesh_d_vals = fpr_dict_kesh[fpr_r]
        our_max = max(our_d_vals)
        cai_max = max(cai_d_vals)
        kesh_max = max(kesh_d_vals)
        glob_max = max([our_max, cai_max, kesh_max])
        if glob_max > ylim_max:
            ylim_max = glob_max
        plot_confidence_interval(x=1, values=our_d_vals, z=1.96, color='#2187bb', horizontal_line_width=0.25)
        plot_idx_counter += 1
        plot_confidence_interval(x=2, values=cai_d_vals, z=1.96, color='#2187bb', horizontal_line_width=0.25)
        plot_idx_counter += 1
        plot_confidence_interval(x=3, values=kesh_d_vals, z=1.96, color='#2187bb', horizontal_line_width=0.25)
        plot_idx_counter += 1
        plt.ylim(0.0, ylim_max)
        plt.title("Confidence Intervals {} Storm".format(storm_name))
        plt.ylabel("Detect Time (5min Intervals)")
        plt.xlabel("FPR")
        plt.tight_layout()
        plt.savefig(os.path.join(curr_storm_save_path, 'conf_avg_amoc_{}_{}.png'.format(storm_name, fpr_r)))
        plt.close()

    threshold_idx_mask = np.where(thresholds_ours <= 0.2) # THESE ARE IDENTICAL INDICES ACROSS THE ALGORITHMS
    means_per_threshold_ours = means_per_threshold_ours[threshold_idx_mask]
    means_per_threshold_cai = means_per_threshold_cai[threshold_idx_mask]
    means_per_threshold_kesh = means_per_threshold_kesh[threshold_idx_mask]
    thresholds_ours = thresholds_ours[threshold_idx_mask]
    thresholds_cai = thresholds_cai[threshold_idx_mask]
    thresholds_kesh = thresholds_kesh[threshold_idx_mask]

    #detect_idx_mask_ours = np.where(means_per_threshold_ours <= 200) # maximum detection time, 2*window_size
    #detect_idx_mask_cai = np.where(means_per_threshold_cai <= 200)
    #detect_idx_mask_kesh = np.where(means_per_threshold_kesh <= 200)
    #detect_idx_mask = reduce(np.intersect1d, (detect_idx_mask_ours, detect_idx_mask_cai, detect_idx_mask_kesh)) # union of all idxs
    #detect_idx_mask = reduce(np.intersect1d, (detect_idx_mask_ours, detect_idx_mask_kesh)) # union of all idxs
    
    #means_per_threshold_ours = means_per_threshold_ours[detect_idx_mask] # 4 seems fine
    #means_per_threshold_cai = means_per_threshold_cai[detect_idx_mask]
    #means_per_threshold_kesh = means_per_threshold_kesh[detect_idx_mask]
    #thresholds_ours = thresholds_ours[detect_idx_mask]
    #thresholds_cai = thresholds_cai[detect_idx_mask]
    #thresholds_kesh = thresholds_kesh[detect_idx_mask]


    """
    ###########################
    """
    plt.plot(thresholds_ours, means_per_threshold_ours, '.b-', label='Ours')
    plt.plot(thresholds_cai, means_per_threshold_cai, '.r-', label='Cai')
    plt.plot(thresholds_kesh, means_per_threshold_kesh, '.g-', label='Kesh')
    plt.xlabel('FPR')
    plt.ylabel('Detection Time (5min Intervals)')
    plt.title('Average AMOC Curve for {} Storm'.format(storm_name))
    plt.legend(loc='best')
    plt.ylim(0.0, max(means_per_threshold_kesh.max(), means_per_threshold_ours.max()))
    
    if not os.path.exists(curr_storm_save_path):
        os.mkdir(curr_storm_save_path)
    plt.savefig(os.path.join(curr_storm_save_path, 'avg_amoc_{}.png'.format(storm_name)))
    plt.close()


        


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sims', action='store_true')
    parser.add_argument('--mesonet', action='store_true')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()
    if args.sims:
        main_sims()
    if args.mesonet:
        main_mesonet()
