import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import chi2, multivariate_normal, zscore
from tqdm import tqdm
from scipy.stats import norm, wilcoxon
from numba import jit
import statistics
from math import sqrt
import argparse
from sklearn.metrics import auc
sns.set()
#sns.set_style('dark')
sns.set_palette("viridis")
import matplotlib

from functools import reduce
import statistics

matplotlib.use('Agg')


def plot_confidence_interval(x, values, z=1.96, color='#2187bb', horizontal_line_width=0.25, constrain=False):
    mean = statistics.mean(values)
    median = statistics.median(values)
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
    #plt.plot(x, median, 'o', color='yellow')
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


def average_AMOC_new(fprs, dts, debug_title="Ours", to_plot=True):
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
        if thresh == 0.05 and to_plot==True:
            np.savetxt('debugging_figs/detect_times_0.05_{}.csv'.format(debug_title), detect_times_per_thresh, delimiter=',')
        elif thresh == 0.01 and to_plot==True:
            np.savetxt('debugging_figs/detect_times_0.01_{}.csv'.format(debug_title), detect_times_per_thresh, delimiter=',')
    
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

# def amoc_gen_vectorized(lrt_values, first_detect_time, last_detect_time, max_time_detection, thresholds=None, p_values=False):
#     if not thresholds:
#         thresholds = sorted(list(lrt_values))

#     fpr_array = []
#     detect_array = []

#     for thresh in thresholds:
#         if p_values: # if we're using p-values, lower is the cutoff
#             exceed_mask = (lrt_values <= thresh)
#         else: # if we're using LRT-values, higher is the cutoff
#             exceed_mask = (lrt_values >= thresh)

#         # calculate false_positives
#         early_fp = np.sum(exceed_mask[0:first_detect_time])
#         late_fp = np.sum(exceed_mask[last_detect_time+1:])

#         # calculate true_negatives
#         early_tn = np.sum(~exceed_mask[0:first_detect_time])
#         late_tn = np.sum(~exceed_mask[last_detect_time+1:])

#         cp_slice = lrt_values[first_detect_time:last_detect_time]
#         cp_mask_slice = exceed_mask[first_detect_time:last_detect_time]
#         # we didn't find it
#         if np.sum(cp_mask_slice) < 1:
#             detect_array.append(max_time_detection)
#         # we found it, take the earliest True
#         else:
#             earliest_time = np.where(cp_mask_slice==True)[0][0]
#             #print(earliest_time)
#             detect_array.append(earliest_time) # earliest detection
        
#         false_positives = early_fp + late_fp
#         true_negatives = early_tn + late_tn
#         if false_positives + true_negatives < 1: # shouldn't happen, but if everything is storm
#             fp_rate = 0.0
#         else:
#             fp_rate = false_positives/(false_positives+true_negatives)
#         fpr_array.append(fp_rate)

#     return fpr_array, detect_array

def amoc_gen_vectorized(lrt_values, first_detect_time, last_detect_time, max_time_detection, thresholds=None, p_values=False, all_fpr=True):
    if not thresholds:
        if p_values:
            thresholds = sorted(list(lrt_values), reverse=False)
        else:
            thresholds = sorted(list(lrt_values), reverse=True)

    fpr_array = []
    detect_array = []

    for thresh in thresholds:
        if p_values: # if we're using p-values, lower is the cutoff
            exceed_mask = (lrt_values <= thresh)
            lrt_mask = (lrt_values[first_detect_time:last_detect_time] <= thresh)
        else: # if we're using LRT-values, higher is the cutoff
            exceed_mask = (lrt_values >= thresh)
            lrt_mask = (lrt_values[first_detect_time:last_detect_time] >= thresh)

        # calculate false_positives
        early_fp = np.sum(exceed_mask[0:first_detect_time])

        earliest_alarm = np.argmax(exceed_mask)
        fp_rate = 0.0
        detection_time = max_time_detection
        if early_fp > 0: # case of false positives
            #detection_time = earliest_alarm # + penalty
            correct_detections = np.sum(exceed_mask[first_detect_time:last_detect_time])
            if correct_detections > 0.0:
                detection_time = exceed_mask[first_detect_time:last_detect_time].argmax() + 1
                fp_rate = early_fp
            else: # you missed it
                detection_time = max_time_detection
                fp_rate = early_fp
            # if all_fpr:
            #     fp_rate = false_positives/(first_detect_time) # total FP over the number of days until storm
            # else:
            #     fp_rate = np.sum(exceed_mask[0:earliest_alarm+1])/(earliest_alarm+1) # FP up til first alarm
            # if fp_rate > 1:
            #     print("FPR", fp_rate)
            #     print("FPS", false_positives)
            #     print("Earliest Alarm", earliest_alarm + 1)
            #     exit()
            #detection_time = max_time_detection
        elif earliest_alarm >= first_detect_time and earliest_alarm < last_detect_time: # no false positives, calculate from detection window
            #detection_time = exceed_mask[first_detect_time:last_detect_time].argmax()
            detection_time = earliest_alarm - first_detect_time + 1
            fp_rate = early_fp
        else: # you missed it
            detection_time = max_time_detection
            fp_rate = early_fp
        
        fpr_array.append(fp_rate)
        detect_array.append(detection_time)

    fpr_array = (np.array(fpr_array)/np.max(fpr_array)).tolist()
    #print(fpr_array)
    #print(detect_array)
    assert np.all(np.diff(np.array(fpr_array)) >= 0)
    assert np.all(np.diff(np.array(detect_array)) <= 0), print(fpr_array, detect_array)
    
    return fpr_array, detect_array

# def amoc_gen_vectorized_fawcett(lrt_values, first_detect_time, last_detect_time, max_time_detection, thresholds=None, p_values=False, all_fpr=True):
#     if not thresholds:
#         if p_values:
#             thresholds = sorted(list(lrt_values), reverse=True)
#         else:
#             thresholds = sorted(list(lrt_values), reverse=False)

#     fpr_array = []
#     score_array = []
#     merged_list = []
#     for thresh in thresholds:
#         F = 0
#         S = 0
#         H = None # first alarm tracker
#         fpr_R = []
#         score_R = []
#         R = []
#         for i, datapoint in enumerate(lrt_values):
#             if p_values: # if we're using p-values, lower is the cutoff
#                 exceeded = (datapoint <= thresh)
#             else: # if we're using LRT-values, higher is the cutoff
#                 exceeded = (datapoint >= thresh)
#             if exceeded:
#                 #print("EXCEEDED")
#                 if i < first_detect_time: # false alarm
#                     F = F + 1
#                 else: # in activity period
#                     if H is None: # first alarm
#                         detection_time = first_detect_time - i + 1
#                         score = max_time_detection - detection_time
#                         S = S + score  # maximum detection time - current index - first detection possibility
#                         H = i
#                     elif i < H:
#                         detection_time = first_detect_time - i + 1
#                         detection_time_H = first_detect_time - H + 1
#                         score = max_time_detection - detection_time
#                         score_H = max_time_detection - detection_time_H
#                         S = S - score_H
#                         S = S + score
#                         H = i
#             fpr_R.append(F)
#             score_R.append(S)
#             R.append((F, S))
#         S_total = S
#         F_total = F
#         if F_total > 0:
#             fpr_R = np.array(fpr_R)/F_total
#             #fpr_R = fpr_R.tolist()
#         else:
#             fpr_R = np.array(fpr_R)
#         if S_total > 0:
#             score_R = np.array(score_R)/S_total
#         else:
#             score_R = np.array(score_R)
#         #score_R= score_R.tolist()
#         new_R = []
#         for fpr, score in R:
#             if F_total > 0:
#                 fpr = fpr/F_total
#             if S_total > 0:
#                 score = score/F_total
#             new_R.append((fpr, score))
#         merged_list.append(new_R)
#         fpr_array.append(fpr_R)
#         score_array.append(score_R)
#     print(fpr_array, score_array)
#     exit()
#     means_per_threshold, thresholds_, fpr_dict = average_AMOC_new(fprs=fpr_array, dts=score_array, debug_title='testing', to_plot=False)
    
#     plt.plot(thresholds_, means_per_threshold)
#     plt.savefig("debugging_figs/AMOC_TEST.png")
#     plt.close()
#     exit()
    


#     #print(fpr_array)
#     #print(detect_array)
#     #assert np.all(np.diff(np.array(fpr_array)) >= 0)
#     #assert np.all(np.diff(np.array(detect_array)) <= 0), print(fpr_array, detect_array)
    
#     return fpr_array, score_array



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
    
    # fprs, detections = amoc_gen_vectorized_fawcett(lrt_vals, 
    #                                 first_detect_time=first_possible_detect_time, 
    #                                 last_detect_time=last_possible_detect_time, 
    #                                 max_time_detection=last_possible_detect_time-first_possible_detect_time,
    #                                 p_values=p_values
    #                                 )

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

def main_sims_avesnov(): # changes the window calculations because it slides both of them
    curr_path = os.getcwd()
    save_path = os.path.join(curr_path, 'amoc_figs/')
    seeds = np.arange(50, 70)
    #seeds = np.arange(50, 54)
    #sim_types = ['anderson_residual_block', 'anderson_residual_unstructured', 'orthogonal']
    #sim_types = ['anderson_residual_unstructured']
    sim_types = ['orthogonal_block_fix_window']
    #sim_types = ['cai_model_one_fix_window']
    #sim_types = ['cai_model_one']
    dims = [20, 40, 60, 80]
    #dims = [20,40,60]
    #dims = [80]
    #dims = [24, 30, 36, 42, 48, 56, 64]
    #dims = [20, 24, 30]
    #dims=[36]
    our_results = {}
    avesenov_results = {}
    for curr_dim in dims:
        curr_dim_path = os.path.join(save_path, str(curr_dim))
        if curr_dim == 20:
            window_size = 50
        elif curr_dim == 40:
            window_size = 100
        elif curr_dim == 60:
            window_size = 150
        elif curr_dim == 80:
            window_size = 200
        if not os.path.exists(curr_dim_path):
            os.mkdir(curr_dim_path)
        for sim_type in sim_types:
            print(sim_type)
            our_seed_dict = {}
            avesenov_seed_dict = {}
            fpr_detect_pairs_ours = []
            fpr_detect_pairs_avesenov = []
            lol_ours_fprs = []
            lol_ours_dts = []
            lol_avesenov_fprs = []
            lol_avesenov_dts = []
            statset_ours = []
            statset_avesenov = []
            for curr_seed in seeds:
                cpd_path='./results/simulation_results/{}_{}/{}/lrt_vals.csv'.format(sim_type, curr_dim, curr_seed)
                avesenov_path='./results/simulation_results_avesenov/{}_{}/{}/global_test_vals.csv'.format(sim_type, curr_dim, curr_seed)
                cpd_vals = np.loadtxt(cpd_path, delimiter=',')
                cutoff = cpd_vals.shape[1]//2
                pvals_cpd = cpd_vals[:, cutoff:] # just the p-values
                stats_cpd = cpd_vals[:, 0:cutoff] # just the stats
                avesenov_cpd = np.loadtxt(avesenov_path, delimiter=',')[:-1]
                cp_location = 500 # THIS IS THE ACTUAL LOCATION, FIRST DETECTION TIME IS (ACTUAL LOCATION - WINDOW SIZE + 1)
                first_d_time = cp_location-window_size+1
                last_d_time = cp_location + window_size
                pvals_cpd = pvals_cpd.min(axis=1)
                
                all_thresholds = None
                
                lrt_result_cpd, detect_result_cpd = amoc_lrt_vals(pvals_cpd, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=None, p_values=True)

                lrt_result_avesenov, detect_result_avesenov = amoc_lrt_vals(avesenov_cpd, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=None, p_values=False)

                our_seed_dict[curr_seed] = (lrt_result_cpd, detect_result_cpd)
                avesenov_seed_dict[curr_seed] = (lrt_result_avesenov, detect_result_avesenov)
                our_results[sim_type] = our_seed_dict
                avesenov_results[sim_type] = avesenov_seed_dict
                print(pvals_cpd.shape, stats_cpd.shape, stats_cpd.max(axis=1).shape, avesenov_cpd.shape)

                sorted_lrt_result_cpd = lrt_result_cpd #sorted(lrt_result_cpd, reverse=True)
                sorted_lrt_result_avesenov = lrt_result_avesenov #sorted(lrt_result_cai, reverse=True)
                sorted_detect_result_cpd = detect_result_cpd #sorted(detect_result_cpd, reverse=False)
                sorted_detect_result_avesenov = detect_result_avesenov #sorted(detect_result_kesh, reverse=False)

                plt.step(sorted_lrt_result_cpd, sorted_detect_result_cpd, '-.b', 
                        label='LD-CPD', where='post')
                plt.step(sorted_lrt_result_avesenov, sorted_detect_result_avesenov, '-.m', 
                        label='Cai', where='post')
                plt.ylabel("Detect Time")
                plt.xlabel("FPR")
                plt.title("{} Seed {}".format(sim_type, curr_seed))
                plt.legend(loc='best')
                plt.savefig(os.path.join(curr_dim_path, 'amoc_{}_seed_{}_dim_{}.png'.format(sim_type, curr_seed, curr_dim)))
                plt.close()


                merged_list_ours = [(sorted_lrt_result_cpd[i], 
                                    sorted_detect_result_cpd[i]) for i in range(0, len(lrt_result_cpd))]
                merged_list_avesenov = [(sorted_lrt_result_avesenov[i], 
                                    sorted_detect_result_avesenov[i]) for i in range(0, len(lrt_result_avesenov))]
                fpr_detect_pairs_ours.append(merged_list_ours)
                fpr_detect_pairs_avesenov.append(merged_list_avesenov)

                # list-of-list appending
                lol_ours_fprs.append(sorted_lrt_result_cpd)
                lol_ours_dts.append(sorted_detect_result_cpd)
                lol_avesenov_fprs.append(sorted_lrt_result_avesenov)
                lol_avesenov_dts.append(sorted_detect_result_avesenov)
                
                statset_ours.append(stats_cpd.flatten())
                statset_avesenov.append(avesenov_cpd.flatten())
            
            means_per_threshold_ours, thresholds_ours, fpr_dict_ours = average_AMOC_new(fprs=lol_ours_fprs, dts=lol_ours_dts)
            means_per_threshold_avesenov, thresholds_avesenov, fpr_dict_avesenov = average_AMOC_new(fprs=lol_avesenov_fprs, dts=lol_avesenov_dts)

            plt.xticks([1, 2, 3, 4, 5, 6], ["0.01_ours", "0.01_avesenov", "0.05_ours", "0.05_avesenov","0.1_ours", "0.1_avesenov"])
            plt.xticks(rotation=45, ha='right')
            plot_idx_counter = 1
            ylim_max = 0.0
            for fpr_r in [0.01, 0.05, 0.1]:
                our_d_vals = fpr_dict_ours[fpr_r]
                avesenov_d_vals = fpr_dict_avesenov[fpr_r]
                our_max = max(our_d_vals)
                avesenov_max = max(avesenov_d_vals)
                glob_max = max([our_max, avesenov_max])
                if glob_max > ylim_max:
                    ylim_max = glob_max
                plot_confidence_interval(x=plot_idx_counter, values=our_d_vals, z=1.96, color='#2187bb', horizontal_line_width=0.25)
                plot_idx_counter += 1
                plot_confidence_interval(x=plot_idx_counter, values=avesenov_d_vals, z=1.96, color='#2187bb', horizontal_line_width=0.25)
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
            threshold_idx_mask = np.where(thresholds_ours <= 0.05) # THESE ARE IDENTICAL INDICES ACROSS THE ALGORITHMS
            means_per_threshold_ours = means_per_threshold_ours[threshold_idx_mask]
            means_per_threshold_avesenov = means_per_threshold_avesenov[threshold_idx_mask]
            thresholds_ours = thresholds_ours[threshold_idx_mask]
            thresholds_avesenov = thresholds_avesenov[threshold_idx_mask]

            #detect_idx_mask_ours = np.where(means_per_threshold_ours <= 200) # maximum detection time, 2*window_size
            #detect_idx_mask_cai = np.where(means_per_threshold_cai <= 200)
            #detect_idx_mask_kesh = np.where(means_per_threshold_kesh <= 200)
            #detect_idx_mask = reduce(np.intersect1d, (detect_idx_mask_ours, detect_idx_mask_cai, detect_idx_mask_kesh)) # union of all idxs
            
            # means_per_threshold_ours = means_per_threshold_ours[detect_idx_mask] # 4 seems fine
            # means_per_threshold_cai = means_per_threshold_cai[detect_idx_mask]
            # means_per_threshold_kesh = means_per_threshold_kesh[detect_idx_mask]
            # thresholds_ours = thresholds_ours[detect_idx_mask]
            # thresholds_cai = thresholds_cai[detect_idx_mask]
            # thresholds_kesh = thresholds_kesh[detect_idx_mask]


            """
            ###########################
            """
            plt.figure(figsize=(15, 12))
            plt.plot(thresholds_ours, means_per_threshold_ours, color='blue', linestyle='-.', label='LD-CPD', linewidth=2.5)
            plt.plot(thresholds_avesenov, means_per_threshold_avesenov, color='magenta', linestyle='-.', label='Avanesov', linewidth=2.5)
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            plt.xlabel('FPR', fontsize=34)
            plt.ylabel('Detection Time', fontsize=34)
            #plt.title('Average AMOC Curve for {} Dim {}'.format(sim_type, curr_dim))
            plt.title('Average AMOC Curve for {} Dim {} vs Avanesov'.format("Block Matrix", curr_dim), fontsize=30)
            plt.ylim(bottom=-5.0)
            #plt.tight_layout()
            plt.legend(loc='best', fontsize=30)
            plt.savefig(os.path.join(curr_dim_path, 'avanesov/avg_amoc_{}_dim_{}_{}.png'.format(sim_type, curr_dim, "avanesov")))
            plt.close()



def main_sims(clime=False):
    curr_path = os.getcwd()
    save_path = os.path.join(curr_path, 'amoc_figs/')
    seeds = np.arange(51, 70)
    #sim_types = ['anderson_residual_block', 'anderson_residual_unstructured', 'orthogonal']
    #sim_types = ['anderson_residual_unstructured']
    sim_types = ['orthogonal_small']
    #sim_types = ['orthogonal_cross_block']
    #sim_types = ['orthogonal_multiple_block']
    # sim_types = ['orthogonal_hard']
    #sim_types = ['orthogonal_cross_hard']
    # sim_types = ['cai_model_one', 'cai_model_three']
    # sim_types = ['orthogonal_small', 'orthogonal_cross_block', 'orthogonal_multiple_block', 'orthogonal_hard', 'cai_model_one']
    # sim_types = ['orthogonal_cross_block', 'orthogonal_multiple_block', 'orthogonal_hard', 'cai_model_one']
    dims = [20, 40, 60, 80]
    #dims = [20,40,60]
    #dims = [80]
    #dims = [24, 30, 36, 42, 48, 56, 64]
    #dims = [20, 24, 30]
    #dims=[36]
    post_window_size = 20
    our_results = {}
    cai_results = {}
    kesh_results = {}
    kesh_alt_results = {}
    for curr_dim in dims:
        curr_dim_path = os.path.join(save_path, str(curr_dim))
        if curr_dim == 20:
            window_size = 50
        elif curr_dim == 40:
            window_size = 100
        elif curr_dim == 60:
            window_size = 150
        elif curr_dim == 80:
            window_size = 200
        if not os.path.exists(curr_dim_path):
            os.mkdir(curr_dim_path)
        for sim_type in sim_types:
            print(sim_type)
            our_seed_dict = {}
            cai_seed_dict = {}
            kesh_seed_dict = {}
            kesh_alt_seed_dict = {}
            fpr_detect_pairs_ours = []
            fpr_detect_pairs_cai = []
            fpr_detect_pairs_kesh = []
            fpr_detect_pairs_kesh_alt = []
            lol_ours_fprs = []
            lol_ours_dts = []
            lol_cai_fprs = []
            lol_cai_dts = []
            lol_kesh_fprs = []
            lol_kesh_dts = []
            lol_kesh_alt_fprs = []
            lol_kesh_alt_dts = []
            statset_ours = []
            statset_cai = []
            statset_kesh = []
            statset_kesh_alt = []
            auc_ours = []
            auc_kesh = []
            auc_kesh_alt = []
            auc_cai = []
            for curr_seed in seeds:
                cpd_path='./results/simulation_results/{}_{}/{}/lrt_vals.csv'.format(sim_type, curr_dim, curr_seed)
                xia_path='./results/simulation_results_cai/{}_{}/{}/global_test_vals.csv'.format(sim_type, curr_dim, curr_seed)
                kesh_root = './results/simulation_results_kesh'
                kesh_alt_root = './results/simulation_results_kesh_alt'
                if clime:
                    kesh_root = kesh_root + "_clime"
                    kesh_alt_root = kesh_alt_root + "_clime"
                kesh_path='{}/{}_{}/{}/global_test_vals.csv'.format(kesh_root, sim_type, curr_dim, curr_seed)
                kesh_path_alt='{}/{}_{}/{}/global_test_vals.csv'.format(kesh_alt_root, sim_type, curr_dim, curr_seed)
                cpd_vals = np.loadtxt(cpd_path, delimiter=',')
                cutoff = cpd_vals.shape[1]//2
                pvals_cpd = cpd_vals[:, cutoff:][window_size:] # just the p-values
                stats_cpd = cpd_vals[:, 0:cutoff][window_size:] # just the stats
                xia_cpd = np.loadtxt(xia_path, delimiter=',')
                kesh_cpd = np.loadtxt(kesh_path, delimiter=',')
                kesh_cpd_alt = np.loadtxt(kesh_path_alt, delimiter=',')
                print("SIM {} DIM {} SEED {} WINDOW {} RAW {} PVALS {} STATS {} XIA {} KESH {} KESH_ALT {}".format(
                    sim_type, curr_dim, curr_seed, window_size,
                    cpd_vals.shape, pvals_cpd.shape, stats_cpd.shape,
                    xia_cpd.shape, kesh_cpd.shape, kesh_cpd_alt.shape))
                #pvals_cpd_padded = np.pad(pvals_cpd, [(100, 100), (0,0)], mode='constant')
                #stats_cpd_padded = np.pad(stats_cpd, [(100, 100), (0,0)], mode='constant')[0:-100, :]
                #stats_cpd = stats_cpd_padded
                #xia_cpd_padded = np.pad(xia_cpd, 100, mode='constant')
                #cp_location = stats_cpd_padded.shape[0]//2
                cp_location = 500 - window_size # THIS IS THE ACTUAL LOCATION, FIRST DETECTION TIME IS (ACTUAL LOCATION - WINDOW SIZE + 1)
                first_d_time = cp_location-post_window_size+1
                last_d_time = len(pvals_cpd) - 1
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
                
                lrt_result_cpd, detect_result_cpd = amoc_lrt_vals(pvals_cpd, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=None, p_values=True)
                #lrt_result_cai, detect_result_cai = amoc_lrt_vals(pvals_cai, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=all_thresholds, p_values=True)
                #lrt_result_kesh, detect_result_kesh = amoc_lrt_vals(pvals_kesh, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=all_thresholds, p_values=True)

                #lrt_result_cpd, detect_result_cpd = amoc_lrt_vals(stats_cpd.max(axis=1), first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=None, p_values=False)
                lrt_result_cai, detect_result_cai = amoc_lrt_vals(xia_cpd, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=None, p_values=False)
                lrt_result_kesh, detect_result_kesh = amoc_lrt_vals(kesh_cpd, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=None, p_values=False)
                lrt_result_kesh_alt, detect_result_kesh_alt = amoc_lrt_vals(kesh_cpd_alt, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=None, p_values=False)

                cutidx_ours = np.where(lrt_result_cpd <= 0.05)
                cutidx_kesh = np.where(lrt_result_kesh <= 0.05)
                cutidx_kesh_alt = np.where(lrt_result_kesh_alt <= 0.05)
                cutidx_cai = np.where(lrt_result_cai <= 0.05)
                auc_ours.append(auc(x=lrt_result_cpd[cutidx_ours], y=detect_result_cpd[cutidx_ours]))
                auc_kesh.append(auc(x=lrt_result_kesh[cutidx_kesh], y=detect_result_kesh[cutidx_kesh]))
                auc_kesh_alt.append(auc(x=lrt_result_kesh_alt[cutidx_kesh_alt], y=detect_result_kesh_alt[cutidx_kesh_alt]))
                auc_cai.append(auc(x=lrt_result_cai[cutidx_cai], y=detect_result_cai[cutidx_cai]))

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
                kesh_alt_seed_dict[curr_seed] = (lrt_result_kesh_alt, detect_result_kesh_alt)
                our_results[sim_type] = our_seed_dict
                cai_results[sim_type] = cai_seed_dict
                kesh_results[sim_type] = kesh_seed_dict
                kesh_alt_results[sim_type] = kesh_alt_seed_dict
                print("SIM {} DIM {} SEED {} WINDOW {} PVALS {} STATS {} STATS_MAX {} XIA {} KESH {} KESH_ALT {}".format(
                    sim_type, curr_dim, curr_seed, window_size,
                    pvals_cpd.shape, stats_cpd.shape, stats_cpd.max(axis=1).shape,
                    xia_cpd.shape, kesh_cpd.shape, kesh_cpd_alt.shape))

                sorted_lrt_result_cpd = lrt_result_cpd #sorted(lrt_result_cpd, reverse=True)
                sorted_lrt_result_cai = lrt_result_cai #sorted(lrt_result_cai, reverse=True)
                sorted_lrt_result_kesh = lrt_result_kesh #sorted(lrt_result_kesh, reverse=True)
                sorted_lrt_result_kesh_alt = lrt_result_kesh_alt
                sorted_detect_result_cpd = detect_result_cpd #sorted(detect_result_cpd, reverse=False)
                sorted_detect_result_cai = detect_result_cai #sorted(detect_result_cai, reverse=False)
                sorted_detect_result_kesh = detect_result_kesh#sorted(detect_result_kesh, reverse=False)
                sorted_detect_result_kesh_alt = detect_result_kesh_alt 

                plt.step(sorted_lrt_result_cpd, sorted_detect_result_cpd, '-.b', 
                        label='LD-CPD', where='post')
                plt.step(sorted_lrt_result_kesh, sorted_detect_result_kesh, '-.g', 
                        label='Kesh', where='post')
                plt.step(sorted_lrt_result_kesh_alt, sorted_detect_result_kesh_alt, '-.k', 
                        label='KeshAlt', where='post')
                plt.step(sorted_lrt_result_cai, sorted_detect_result_cai, '-.r', 
                        label='Cai', where='post')
                plt.ylabel("Detect Time")
                plt.xlabel("FPR")
                plt.title("{} Seed {}".format(sim_type, curr_seed))
                plt.legend(loc='best')
                amoc_suffix = "_clime" if clime else ""
                plt.savefig(os.path.join(curr_dim_path, 'amoc_{}_seed_{}_dim_{}{}.png'.format(sim_type, curr_seed, curr_dim, amoc_suffix)))
                plt.close()


                merged_list_ours = [(sorted_lrt_result_cpd[i], 
                                    sorted_detect_result_cpd[i]) for i in range(0, len(lrt_result_cpd))]
                merged_list_cai = [(sorted_lrt_result_cai[i], 
                                    sorted_detect_result_cai[i]) for i in range(0, len(lrt_result_cai))]
                merged_list_kesh = [(sorted_lrt_result_kesh[i], 
                                    sorted_detect_result_kesh[i]) for i in range(0, len(lrt_result_kesh))]
                merged_list_kesh_alt = [(sorted_lrt_result_kesh_alt[i], 
                                    sorted_detect_result_kesh_alt[i]) for i in range(0, len(lrt_result_kesh_alt))]
                fpr_detect_pairs_ours.append(merged_list_ours)
                fpr_detect_pairs_cai.append(merged_list_cai)
                fpr_detect_pairs_kesh.append(merged_list_kesh)
                fpr_detect_pairs_kesh_alt.append(merged_list_kesh_alt)

                # list-of-list appending
                lol_ours_fprs.append(sorted_lrt_result_cpd)
                lol_ours_dts.append(sorted_detect_result_cpd)
                lol_cai_fprs.append(sorted_lrt_result_cai)
                lol_cai_dts.append(sorted_detect_result_cai)
                lol_kesh_fprs.append(sorted_lrt_result_kesh)
                lol_kesh_dts.append(sorted_detect_result_kesh)
                lol_kesh_alt_fprs.append(sorted_lrt_result_kesh_alt)
                lol_kesh_alt_dts.append(sorted_detect_result_kesh_alt)
                
                statset_ours.append(stats_cpd.flatten())
                statset_cai.append(xia_cpd.flatten())
                statset_kesh.append(kesh_cpd.flatten())
                statset_kesh_alt.append(kesh_cpd_alt.flatten())
            
            #means_per_threshold_ours, thresholds_ours, fpr_dict_ours = average_AMOC(fpr_detect_pairs_ours)
            #means_per_threshold_cai, thresholds_cai, fpr_dict_cai = average_AMOC(fpr_detect_pairs_cai)
            #means_per_threshold_kesh, thresholds_kesh, fpr_dict_kesh = average_AMOC(fpr_detect_pairs_kesh)
            means_per_threshold_ours, thresholds_ours, fpr_dict_ours = average_AMOC_new(fprs=lol_ours_fprs, dts=lol_ours_dts, debug_title='Ours_Block_{}'.format(curr_dim))
            means_per_threshold_cai, thresholds_cai, fpr_dict_cai = average_AMOC_new(fprs=lol_cai_fprs, dts=lol_cai_dts, debug_title='Cai_Block_{}'.format(curr_dim))
            means_per_threshold_kesh, thresholds_kesh, fpr_dict_kesh = average_AMOC_new(fprs=lol_kesh_fprs, dts=lol_kesh_dts, debug_title='KMA_Block_{}'.format(curr_dim))
            means_per_threshold_kesh_alt, thresholds_kesh_alt, fpr_dict_kesh_alt = average_AMOC_new(fprs=lol_kesh_alt_fprs, dts=lol_kesh_alt_dts, debug_title='KM_Block_{}'.format(curr_dim))

            # plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9], ["0.01_ours", "0.01_cai", "0.01_kesh", "0.05_ours", "0.05_cai", "0.05_kesh", "0.1_ours", "0.1_cai", "0.1_kesh"])
            # plt.xticks(rotation=45, ha='right')
            # plot_idx_counter = 1
            # ylim_max = 0.0
            # for fpr_r in [0.01, 0.05, 0.1]:
            #     our_d_vals = fpr_dict_ours[fpr_r]
            #     cai_d_vals = fpr_dict_cai[fpr_r]
            #     kesh_d_vals = fpr_dict_kesh[fpr_r]
            #     our_max = max(our_d_vals)
            #     cai_max = max(cai_d_vals)
            #     kesh_max = max(kesh_d_vals)
            #     glob_max = max([our_max, cai_max, kesh_max])
            #     if glob_max > ylim_max:
            #         ylim_max = glob_max
            #     plot_confidence_interval(x=plot_idx_counter, values=our_d_vals, z=1.96, color='#2187bb', horizontal_line_width=0.25)
            #     plot_idx_counter += 1
            #     plot_confidence_interval(x=plot_idx_counter, values=cai_d_vals, z=1.96, color='#2187bb', horizontal_line_width=0.25)
            #     plot_idx_counter += 1
            #     plot_confidence_interval(x=plot_idx_counter, values=kesh_d_vals, z=1.96, color='#2187bb', horizontal_line_width=0.25)
            #     plot_idx_counter += 1
            # plt.ylim(-10.0, 40.0)
            # plt.title("Confidence Intervals {} Dim {}".format(sim_type, curr_dim))
            # plt.ylabel("Detection Time")
            # plt.xlabel("FPR")
            # plt.tight_layout()
            # plt.savefig(os.path.join(curr_dim_path, 'conf_avg_amoc_{}_dim_{}.png'.format(sim_type, curr_dim)))
            # plt.close()
            
            """
            ###########################
            Filter out highest detection time - unrealistic
            and high FPR - also unrealistic
            """
            threshold_idx_mask = np.where(thresholds_ours <= 0.05) # THESE ARE IDENTICAL INDICES ACROSS THE ALGORITHMS
            means_per_threshold_ours = means_per_threshold_ours[threshold_idx_mask]
            means_per_threshold_cai = means_per_threshold_cai[threshold_idx_mask]
            means_per_threshold_kesh = means_per_threshold_kesh[threshold_idx_mask]
            means_per_threshold_kesh_alt = means_per_threshold_kesh_alt[threshold_idx_mask]
            thresholds_ours = thresholds_ours[threshold_idx_mask]
            thresholds_cai = thresholds_cai[threshold_idx_mask]
            thresholds_kesh = thresholds_kesh[threshold_idx_mask]
            thresholds_kesh_alt = thresholds_kesh_alt[threshold_idx_mask]


            # detect_idx_mask_ours = np.where(means_per_threshold_ours <= 200) # maximum detection time, 2*window_size
            # detect_idx_mask_cai = np.where(means_per_threshold_cai <= 200)
            # detect_idx_mask_kesh = np.where(means_per_threshold_kesh <= 200)
            # detect_idx_mask_kesh_alt = np.where(means_per_threshold_kesh_alt <= 200)
            # detect_idx_mask = reduce(np.intersect1d, (detect_idx_mask_ours, detect_idx_mask_cai, detect_idx_mask_kesh)) # union of all idxs
            
            # means_per_threshold_ours = means_per_threshold_ours[detect_idx_mask] # 4 seems fine
            # means_per_threshold_cai = means_per_threshold_cai[detect_idx_mask]
            # means_per_threshold_kesh = means_per_threshold_kesh[detect_idx_mask]
            # thresholds_ours = thresholds_ours[detect_idx_mask]
            # thresholds_cai = thresholds_cai[detect_idx_mask]
            # thresholds_kesh = thresholds_kesh[detect_idx_mask]


            """
            ###########################
            """
            plt.xticks([1, 2, 3, 4], ["LD-CPD", "XCC", "KMA", "KM"])
            plt.xticks(rotation=45, ha='right')
            plot_confidence_interval(x=1, values=auc_ours, color='blue', horizontal_line_width=0.25)
            plot_confidence_interval(x=2, values=auc_cai, color='red', horizontal_line_width=0.25)
            plot_confidence_interval(x=3, values=auc_kesh, color='green', horizontal_line_width=0.25)
            plot_confidence_interval(x=4, values=auc_kesh_alt, color='black', horizontal_line_width=0.25)
            #plt.ylim(-10.0, 40.0)
            plt.ylim(0.0)
            plt.title("95% Confidence Intervals AMOC-AUC {} Dim {}".format("Multiple", curr_dim))
            plt.ylabel("AMOC-AUC")
            plt.xlabel("FPR")
            plt.tight_layout()
            auc_conf_suffix = "_clime" if clime else ""
            plt.savefig(os.path.join(curr_dim_path+'/avg_amoc/', 'auc_conf_{}_dim_{}{}.png'.format(sim_type, curr_dim, auc_conf_suffix)))
            plt.close()

            np.savetxt(os.path.join(curr_dim_path+'/avg_amoc/', 'auc_ours_{}_dim_{}.csv'.format(sim_type, curr_dim)), auc_ours, delimiter=',')
            np.savetxt(os.path.join(curr_dim_path+'/avg_amoc/', 'auc_cai_{}_dim_{}.csv'.format(sim_type, curr_dim)), auc_cai, delimiter=',')
            kesh_suffix = "_clime" if clime else ""
            np.savetxt(os.path.join(curr_dim_path+'/avg_amoc/', 'auc_kesh{}_{}_dim_{}.csv'.format(kesh_suffix, sim_type, curr_dim)), auc_kesh, delimiter=',')
            np.savetxt(os.path.join(curr_dim_path+'/avg_amoc/', 'auc_kesh_alt{}_{}_dim_{}.csv'.format(kesh_suffix, sim_type, curr_dim)), auc_kesh_alt, delimiter=',')
            auc_ours = np.array(auc_ours).mean()
            auc_kesh = np.array(auc_kesh).mean()
            auc_kesh_alt = np.array(auc_kesh_alt).mean()
            auc_cai = np.array(auc_cai).mean()
            print("Plotting")
            plt.figure(figsize=(15, 12))
            # plt.plot(thresholds_ours, means_per_threshold_ours, '.b-', label='LD-CPD', linewidth=5)
            # plt.plot(thresholds_cai, means_per_threshold_cai, '.r-', label='XCC', linewidth=5)
            # plt.plot(thresholds_kesh, means_per_threshold_kesh, '.g-', label='KMA', linewidth=5)
            # plt.plot(thresholds_kesh_alt, means_per_threshold_kesh_alt, '.k-',label='KM', linewidth=5)
            # plt.plot(thresholds_ours, means_per_threshold_ours, '.b-', label='LD-CPD - AUC {}'.format(round(auc_ours, 2)), linewidth=5)
            # plt.plot(thresholds_cai, means_per_threshold_cai, '.r-', label='XCC - AUC {}'.format(round(auc_cai, 2)), linewidth=5)
            # plt.plot(thresholds_kesh, means_per_threshold_kesh, '.g-', label='KMA - AUC {}'.format(round(auc_kesh, 2)), linewidth=5)
            # plt.plot(thresholds_kesh_alt, means_per_threshold_kesh_alt, '.k-', label='KM - AUC {}'.format(round(auc_kesh_alt, 2)), linewidth=5)
            plt.plot(thresholds_ours, means_per_threshold_ours, '.b-', label='LD-CPD', linewidth=5)
            plt.plot(thresholds_cai, means_per_threshold_cai, '.r-', label='XCC', linewidth=5)
            plt.plot(thresholds_kesh, means_per_threshold_kesh, '.g-', label='KMA', linewidth=5)
            plt.plot(thresholds_kesh_alt, means_per_threshold_kesh_alt, '.k-', label='KM', linewidth=5)
            plt.xlabel('FPR', fontsize=38)
            plt.ylabel('Detection Time', fontsize=38)
            #plt.title('Average AMOC Curve for {} Dim {}'.format(sim_type, curr_dim))
            if sim_type == 'cai_model_one':
                plot_title = 'Banded Matrix Change'
            elif sim_type == 'cai_model_three':
                plot_title = 'Scattered Matrix Change'
            elif sim_type == 'orthogonal_small':
                plot_title = 'Single'
            elif sim_type == 'orthogonal_cross_block':
                plot_title = 'Multiple'
            elif sim_type == 'orthogonal_multiple_block':
                plot_title = 'Indiv Coeffs'
            elif sim_type == 'orthogonal_hard':
                plot_title = 'Indiv Coeffs Subsets'
            elif sim_type == 'orthogonal_cross_hard':
                plot_title = 'Multiple Subsets'
            else:
                plot_title = sim_type
            plt.title('Average AMOC Curve ({} Dim={})'.format(plot_title, curr_dim), fontsize=36)
            plt.ylim(bottom=0.0)
            plt.xticks(fontsize=28)
            plt.yticks(fontsize=28)
            plt.legend(loc='best', fontsize=32)
            avg_suffix = "_clime" if clime else ""
            plt.savefig(os.path.join(curr_dim_path+'/avg_amoc/', 'avg_amoc_{}_dim_{}{}.png'.format(sim_type, curr_dim, avg_suffix)))
            plt.close()

def main_mesonet(storm_name='center', clime=False):
    """
    In progress for MesoNet data collection

    Will be more involved. Mainly for CP location identification with storm pairings
    """
    window_size = 600 # this is fixed currently
    post_window_size = 25 # this is fixed currently
    dim = 35 # this is fixed currently
    curr_path = os.getcwd()
    save_path = os.path.join(curr_path, 'amoc_figs/')
    curr_storm_save_path = save_path+"mesonet/"+storm_name
    # cpd_path = 'results/{}_storm_feb_2024'.format(storm_name)
    # kesh_path = 'results_kesh/mesonet_feb_2024'
    # kesh_alt_path = 'results_kesh_alt/mesonet_feb_2024'
    # cai_path = 'results_cai/mesonet_feb_2024'

    cpd_path = 'results/{}_storm'.format(storm_name)

    cpd_path = 'results/{}_storm_unprocessed'.format(storm_name)
    kesh_path = 'results_kesh/mesonet'
    kesh_alt_path = 'results_kesh_alt/mesonet'
    if clime:
        kesh_path = 'results_kesh_clime/mesonet'
        kesh_alt_path = 'results_kesh_alt_clime/mesonet'
    cai_path = 'results_cai/mesonet'
    storm_data_path = '../data/out_{}'.format(storm_name)
    dir_files = os.listdir(cpd_path)
    storm_dir_files = os.listdir(storm_data_path)
    # storm_nums = []
    # for fi in dir_files:
    #     if 'center_storm' in fi:
    #         storm_nums.append(int(fi.split('_')[-1].split('.')[0]))
    storm_nums = [1,3,4,5,6,7,8,9,12,13,14,15,16,17,23,24,25,26,28,32,33,34,35,38]

    storm_files = []
    for num in storm_nums:
        storm_files.append('{}_storm_{}_storm_data.csv'.format(storm_name, num))
    
    our_results = {}
    kesh_results = {}
    fpr_detect_pairs_ours = []
    fpr_detect_pairs_cai = []
    fpr_detect_pairs_kesh = []
    fpr_detect_pairs_kesh_alt = []
    lol_ours_fprs = []
    lol_ours_dts = []
    lol_cai_fprs = []
    lol_cai_dts = []
    lol_kesh_fprs = []
    lol_kesh_dts = []
    lol_kesh_alt_fprs = []
    lol_kesh_alt_dts = []
    unique_tuples = []
    max_detect_times = []
    storm_type_list = []
    storm_duration_list = []
    #storm_nums = [6]
    auc_ours = []
    auc_kesh = []
    auc_kesh_alt = []
    auc_cai = []
    for num in sorted(storm_nums):
        cpd_vals = np.loadtxt(os.path.join(cpd_path, "{}_storm_{}.csv".format(storm_name, num)), delimiter=',')
        cutoff = cpd_vals.shape[1]//2 # split test stats and p-vals
        pvals_cpd = cpd_vals[:, cutoff:][window_size:] # just the p-values, exclude first window size
        stats_cpd = cpd_vals[:, 0:cutoff][window_size:] # just the stats, exclude first window size

        # kesh_vals = np.loadtxt(os.path.join(kesh_path, "{}_storm_{}_final_data.csv_global_test_vals.csv".format(storm_name, num)), delimiter=',')
        # kesh_vals_alt = np.loadtxt(os.path.join(kesh_alt_path, "{}_storm_{}_final_data.csv_global_test_vals.csv".format(storm_name, num)), delimiter=',')
        # cai_vals = np.loadtxt(os.path.join(cai_path, "{}_storm_{}_final_data.csv_global_test_vals.csv".format(storm_name, num)), delimiter=',')

        kesh_vals = np.loadtxt(os.path.join(kesh_path, "{}_storm_storm_{}_unprocessed_data.csv_global_test_vals.csv".format(storm_name, num)), delimiter=',')
        kesh_vals_alt = np.loadtxt(os.path.join(kesh_alt_path, "{}_storm_storm_{}_unprocessed_data.csv_global_test_vals.csv".format(storm_name, num)), delimiter=',')
        cai_vals = np.loadtxt(os.path.join(cai_path, "{}_storm_storm_{}_unprocessed_data.csv_global_test_vals.csv".format(storm_name, num)), delimiter=',')
        # print(cai_vals.shape)
        # print(kesh_vals.shape)
        # print(pvals_cpd.shape)
        # print(kesh_vals_alt.shape)
        print(num)
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
            elif (storm_end_idx - storm_start_idx) >= 600:
                continue
            else:
                unique_tuples.append(curr_tuple)
        cp_location = storm_start_idx
        cp_location_adjustment = cp_location - window_size
        # subtract off window_size to handle the sliding window edges at start/end of data
        first_d_time = cp_location_adjustment - post_window_size + 1 - 12
        storm_duration = storm_end_idx - storm_start_idx + 12*2
        last_d_time = cp_location_adjustment + storm_duration + 1 - post_window_size + 12 # how long is the storm contained in the window as it slides over the data - at least window_size times (right to left)

        adjustment_window_start = first_d_time - 120
        adjustment_window_end = last_d_time
        #first_d_time = 120
        #last_d_time = last_d_time - adjustment_window_start

        print("*** STORM NUMBER {} ***".format(num))
        print("Storm type {}; Duration {}".format(storm_type, storm_duration))
        print("Storm Start {} End {}".format(storm_start_idx, storm_end_idx))
        print("Adj Storm Start {} End {}".format(storm_start_idx-window_size, storm_end_idx-window_size))
        print("Final Data shape {}".format(actual_final_data.shape))
        storm_type_list.append(storm_type)
        storm_duration_list.append(storm_duration)
        max_detect_times.append(last_d_time-first_d_time)
        #print(pvals_cpd.shape, stats_cpd.shape, kesh_vals.shape, actual_final_data.shape)
        
        
        #print(stats_cpd.min(), stats_cpd.max(), pvals_cpd.max())
        pvals_cpd = pvals_cpd.min(axis=1)
        pvals_kesh = kesh_p_value(kesh_vals)
        pvals_cai = cai_p_value(cai_vals, dim)

        """
        Cut out points after storm
        """
        pvals_cpd_uncut = pvals_cpd.copy()
        kesh_vals_uncut = kesh_vals.copy()
        kesh_vals_alt_uncut = kesh_vals_alt.copy()
        cai_vals_uncut = cai_vals.copy()
        pvals_cpd = pvals_cpd[0:last_d_time+1]
        pvals_kesh = pvals_kesh[0:last_d_time+1]
        pvals_cai = pvals_cai[0:last_d_time+1]
        stats_cpd = stats_cpd.max(axis=1)[0:last_d_time+1]
        kesh_vals = kesh_vals[0:last_d_time+1]
        kesh_vals_alt = kesh_vals_alt[0:last_d_time+1]
        cai_vals = cai_vals[0:last_d_time+1]

        # pvals_cpd = pvals_cpd[adjustment_window_start:adjustment_window_end+1]
        # pvals_kesh = pvals_kesh[adjustment_window_start:adjustment_window_end+1]
        # pvals_cai = pvals_cai[adjustment_window_start:adjustment_window_end+1]
        # stats_cpd = stats_cpd.max(axis=1)[adjustment_window_start:adjustment_window_end+1]
        # kesh_vals = kesh_vals[adjustment_window_start:adjustment_window_end+1]
        # kesh_vals_alt = kesh_vals_alt[adjustment_window_start:adjustment_window_end+1]
        # cai_vals = cai_vals[adjustment_window_start:adjustment_window_end+1]
        """
        """
        print(pvals_cpd.shape, kesh_vals.shape, kesh_vals_alt.shape, cai_vals.shape)
        #exit()

        # lrt_result_cpd, detect_result_cpd = amoc_p_vals(pvals_cpd, 
        #                                                 first_possible_detect_time=first_d_time,
        #                                                 last_possible_detect_time=last_d_time,
        #                                                 dim=None, # deprecated
        #                                                 thresholds=all_thresholds
        #                                                 )
        
        lrt_result_cpd, detect_result_cpd = amoc_lrt_vals(pvals_cpd, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=None, p_values=True)
        lrt_result_kesh, detect_result_kesh = amoc_lrt_vals(kesh_vals, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=None, p_values=False)
        lrt_result_kesh_alt, detect_result_kesh_alt = amoc_lrt_vals(kesh_vals_alt, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=None, p_values=False)
        lrt_result_cai, detect_result_cai = amoc_lrt_vals(cai_vals, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=None, p_values=False)


        cutidx_ours = np.where(lrt_result_cpd <= 0.05)
        cutidx_kesh = np.where(lrt_result_kesh <= 0.05)
        cutidx_kesh_alt = np.where(lrt_result_kesh_alt <= 0.05)
        cutidx_cai = np.where(lrt_result_cai <= 0.05)
        auc_ours.append(auc(x=lrt_result_cpd[cutidx_ours], y=detect_result_cpd[cutidx_ours]))
        auc_kesh.append(auc(x=lrt_result_kesh[cutidx_kesh], y=detect_result_kesh[cutidx_kesh]))
        auc_kesh_alt.append(auc(x=lrt_result_kesh_alt[cutidx_kesh_alt], y=detect_result_kesh_alt[cutidx_kesh_alt]))
        auc_cai.append(auc(x=lrt_result_cai[cutidx_cai], y=detect_result_cai[cutidx_cai]))
        
        print("Detect Ours: Min {} Max {} Mean {}".format(detect_result_cpd.min(), detect_result_cpd.max(), detect_result_cpd.mean()))
        print("Detect Kesh: Min {} Max {} Mean {}".format(detect_result_kesh.min(), detect_result_kesh.max(), detect_result_kesh.mean()))
        print("Detect KeshAlt: Min {} Max {} Mean {}".format(detect_result_kesh_alt.min(), detect_result_kesh_alt.max(), detect_result_kesh_alt.mean()))
        print("Detect Cai: Min {} Max {} Mean {}".format(detect_result_cai.min(), detect_result_cai.max(), detect_result_cai.mean()))
        print("Maximum Detect Time {}".format(last_d_time-first_d_time))
        print("**************************")
        print()

        plt.plot(pvals_cpd_uncut)
        plt.axvline(cp_location_adjustment, linestyle='--', color='green')
        kesh_suffix = "_clime" if clime else ""
        plt.savefig('debugging_figs/mesonet_center/{}_{}{}.png'.format(num, "ours", kesh_suffix))
        plt.close()
        plt.plot(cai_vals_uncut)
        plt.axvline(cp_location_adjustment, linestyle='--', color='green')
        plt.savefig('debugging_figs/mesonet_center/{}_{}{}.png'.format(num, "cai", kesh_suffix))
        plt.close()
        plt.plot(kesh_vals_uncut)
        plt.axvline(cp_location_adjustment, linestyle='--', color='green')
        plt.savefig('debugging_figs/mesonet_center/{}_{}{}.png'.format(num, "kma", kesh_suffix))
        plt.close()
        plt.plot(kesh_vals_alt_uncut)
        plt.axvline(cp_location_adjustment, linestyle='--', color='green')
        plt.savefig('debugging_figs/mesonet_center/{}_{}{}.png'.format(num, "km", kesh_suffix))
        plt.close()

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
        sorted_lrt_result_kesh_alt = lrt_result_kesh_alt
        sorted_detect_result_cpd = detect_result_cpd  #sorted(detect_result_cpd, reverse=False) # 
        sorted_detect_result_cai = detect_result_cai #sorted(detect_result_cai, reverse=False)
        sorted_detect_result_kesh = detect_result_kesh  # sorted(detect_result_kesh, reverse=False) #
        sorted_detect_result_kesh_alt = detect_result_kesh_alt

        merged_list_ours = [(sorted_lrt_result_cpd[i], 
                            sorted_detect_result_cpd[i]) for i in range(0, len(lrt_result_cpd))]
        merged_list_cai = [(sorted_lrt_result_cai[i], 
                            sorted_detect_result_cai[i]) for i in range(0, len(lrt_result_cai))]
        merged_list_kesh = [(sorted_lrt_result_kesh[i], 
                            sorted_detect_result_kesh[i]) for i in range(0, len(lrt_result_kesh))]
        merged_list_kesh_alt = [(sorted_lrt_result_kesh_alt[i], 
                            sorted_detect_result_kesh_alt[i]) for i in range(0, len(lrt_result_kesh_alt))]
        
        fpr_detect_pairs_ours.append(merged_list_ours)
        fpr_detect_pairs_cai.append(merged_list_cai)
        fpr_detect_pairs_kesh.append(merged_list_kesh)
        fpr_detect_pairs_kesh_alt.append(merged_list_kesh_alt)

        lol_ours_fprs.append(sorted_lrt_result_cpd)
        lol_ours_dts.append(sorted_detect_result_cpd)
        lol_cai_fprs.append(sorted_lrt_result_cai)
        lol_cai_dts.append(sorted_detect_result_cai)
        lol_kesh_fprs.append(sorted_lrt_result_kesh)
        lol_kesh_dts.append(sorted_detect_result_kesh)
        lol_kesh_alt_fprs.append(sorted_lrt_result_kesh_alt)
        lol_kesh_alt_dts.append(sorted_detect_result_kesh_alt)

        plt.step(sorted_lrt_result_cpd, sorted_detect_result_cpd, '-.b', 
                        label='LD-CPD', where='post')
        plt.step(sorted_lrt_result_kesh, sorted_detect_result_kesh, '-.g', 
                label='KMA', where='post')
        plt.step(sorted_lrt_result_kesh_alt, sorted_detect_result_kesh_alt, '-.k', 
                label='KM', where='post')
        plt.step(sorted_lrt_result_cai, sorted_detect_result_cai, '-.r', 
                label='Cai', where='post')
        plt.ylabel("Detect Time (5min Intervals)")
        plt.xlabel("FPR")
        plt.title("{} Storm_{}".format(storm_name, num))
        plt.legend(loc='best')
        kesh_suffix = "_clime" if clime else ""
        plt.savefig(os.path.join(curr_storm_save_path, 'amoc{}_{}_{}.png'.format(kesh_suffix, storm_name, num)))
        plt.close()
    #means_per_threshold_ours, thresholds_ours, fpr_dict_ours = average_AMOC(fpr_detect_pairs_ours)
    #means_per_threshold_cai, thresholds_cai, fpr_dict_cai = average_AMOC(fpr_detect_pairs_cai)
    #means_per_threshold_kesh, thresholds_kesh, fpr_dict_kesh = average_AMOC(fpr_detect_pairs_kesh)
    means_per_threshold_ours, thresholds_ours, fpr_dict_ours = average_AMOC_new(fprs=lol_ours_fprs, dts=lol_ours_dts, debug_title='Ours_Mesonet')
    means_per_threshold_cai, thresholds_cai, fpr_dict_cai = average_AMOC_new(fprs=lol_cai_fprs, dts=lol_cai_dts, debug_title='Cai_Mesonet')
    means_per_threshold_kesh, thresholds_kesh, fpr_dict_kesh = average_AMOC_new(fprs=lol_kesh_fprs, dts=lol_kesh_dts, debug_title='KMA_Mesonet')
    means_per_threshold_kesh_alt, thresholds_kesh_alt, fpr_dict_kesh_alt = average_AMOC_new(fprs=lol_kesh_alt_fprs, dts=lol_kesh_alt_dts, debug_title='KM_Mesonet')


    sns.boxplot(auc_ours)
    plt.title('{} Storm AUC Ours FPR <= 0.05'.format(storm_name))
    plt.ylabel('Detection Time')
    plt.savefig(os.path.join(curr_storm_save_path, 'auc_ours_{}.png'.format(storm_name)))
    plt.close()
    sns.boxplot(auc_kesh)
    plt.title('{} Storm AUC KMA FPR <= 0.05'.format(storm_name))
    plt.ylabel('Detection Time')
    kesh_suffix = "_clime" if clime else ""
    plt.savefig(os.path.join(curr_storm_save_path, 'auc_kesh{}_{}.png'.format(kesh_suffix, storm_name)))
    plt.close()
    sns.boxplot(auc_kesh_alt)
    plt.title('{} Storm AUC KM FPR <= 0.05'.format(storm_name))
    plt.ylabel('Detection Time')
    plt.savefig(os.path.join(curr_storm_save_path, 'auc_kesh_alt{}_{}.png'.format(kesh_suffix, storm_name)))
    plt.close()
    sns.boxplot(auc_cai)
    plt.title('{} Storm AUC XCC FPR <= 0.05'.format(storm_name))
    plt.ylabel('Detection Time')
    plt.savefig(os.path.join(curr_storm_save_path, 'auc_cai_{}.png'.format(storm_name)))
    plt.close()

    plt.xticks([1, 2, 3, 4], ["LD-CPD", "XCC", "KMA", "KM"])
    plt.xticks(rotation=45, ha='right')
    plot_confidence_interval(x=1, values=auc_ours, color='blue', horizontal_line_width=0.25)
    plot_confidence_interval(x=2, values=auc_cai, color='red', horizontal_line_width=0.25)
    plot_confidence_interval(x=3, values=auc_kesh, color='green', horizontal_line_width=0.25)
    plot_confidence_interval(x=4, values=auc_kesh_alt, color='black', horizontal_line_width=0.25)
    #plt.ylim(-10.0, 40.0)
    plt.ylim(0.0)
    plt.title("95% Confidence Intervals AMOC-AUC {}".format("Central OK"))
    plt.ylabel("AMOC-AUC")
    #plt.xlabel("FPR")
    plt.tight_layout()
    plt.savefig(os.path.join(curr_storm_save_path, 'auc_conf{}_{}.png'.format(kesh_suffix, storm_name)))
    #plt.savefig(os.path.join(curr_dim_path+'/avg_amoc/', 'auc_conf_{}_dim_{}.png'.format(sim_type, curr_dim)))
    plt.close()

    np.savetxt(os.path.join(curr_storm_save_path, 'auc_ours_{}.csv'.format(storm_name)), auc_ours, delimiter=',')
    np.savetxt(os.path.join(curr_storm_save_path, 'auc_cai_{}.csv'.format(storm_name)), auc_cai, delimiter=',')
    np.savetxt(os.path.join(curr_storm_save_path, 'auc_kesh{}_{}.csv'.format(kesh_suffix, storm_name)), auc_kesh, delimiter=',')
    np.savetxt(os.path.join(curr_storm_save_path, 'auc_kesh_alt{}_{}.csv'.format(kesh_suffix, storm_name)), auc_kesh_alt, delimiter=',')
    auc_ours = np.array(auc_ours).mean()
    auc_kesh = np.array(auc_kesh).mean()
    auc_kesh_alt = np.array(auc_kesh_alt).mean()
    auc_cai = np.array(auc_cai).mean()
    
    # plot_idx_counter = 1
    # ylim_max = 0.0
    # for fpr_r in [0.01, 0.05, 0.1]:
    #     plt.xticks([1, 2, 3], ["{}_ours".format(fpr_r), "{}_cai".format(fpr_r), "{}_kesh".format(fpr_r)])
    #     #plt.xticks([1, 2], ["{}_ours".format(fpr_r), "{}_kesh".format(fpr_r)])
    #     plt.xticks(rotation=45, ha='right')
    #     our_d_vals = fpr_dict_ours[fpr_r]
    #     cai_d_vals = fpr_dict_cai[fpr_r]
    #     kesh_d_vals = fpr_dict_kesh[fpr_r]
    #     our_max = max(our_d_vals)
    #     cai_max = max(cai_d_vals)
    #     kesh_max = max(kesh_d_vals)
    #     glob_max = max([our_max, cai_max, kesh_max])
    #     #glob_max = max([our_max, kesh_max])
    #     if glob_max > ylim_max:
    #         ylim_max = glob_max
    #     plot_confidence_interval(x=1, values=our_d_vals, z=1.96, color='#2187bb', horizontal_line_width=0.25)
    #     plot_idx_counter += 1
    #     plot_confidence_interval(x=2, values=cai_d_vals, z=1.96, color='#2187bb', horizontal_line_width=0.25)
    #     plot_idx_counter += 1
    #     plot_confidence_interval(x=3, values=kesh_d_vals, z=1.96, color='#2187bb', horizontal_line_width=0.25)
    #     plot_idx_counter += 1
    #     #plot_confidence_interval(x=2, values=kesh_d_vals, z=1.96, color='#2187bb', horizontal_line_width=0.25)
    #     plt.ylim(0.0, ylim_max)
    #     plt.title("Confidence Intervals {} Storm".format(storm_name))
    #     plt.ylabel("Detect Time (5min Intervals)")
    #     plt.xlabel("FPR")
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(curr_storm_save_path, 'conf_avg_amoc_{}_{}.png'.format(storm_name, fpr_r)))
    #     plt.close()

    threshold_idx_mask = np.where(thresholds_ours <= 0.05) # THESE ARE IDENTICAL INDICES ACROSS THE ALGORITHMS
    means_per_threshold_ours = means_per_threshold_ours[threshold_idx_mask]
    means_per_threshold_cai = means_per_threshold_cai[threshold_idx_mask]
    means_per_threshold_kesh = means_per_threshold_kesh[threshold_idx_mask]
    means_per_threshold_kesh_alt = means_per_threshold_kesh_alt[threshold_idx_mask]
    thresholds_ours = thresholds_ours[threshold_idx_mask]
    thresholds_cai = thresholds_cai[threshold_idx_mask]
    thresholds_kesh = thresholds_kesh[threshold_idx_mask]
    thresholds_kesh_alt = thresholds_kesh_alt[threshold_idx_mask]

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
    plt.figure(figsize=(25, 14))
    # auc_ours = auc(x=thresholds_ours, y=means_per_threshold_ours)
    # auc_cai = auc(x=thresholds_cai, y=means_per_threshold_cai)
    # auc_kesh = auc(x=thresholds_kesh, y=means_per_threshold_kesh)
    # auc_kesh_alt = auc(x=thresholds_kesh_alt, y=means_per_threshold_kesh_alt)
    # plt.plot(thresholds_ours, means_per_threshold_ours, '.b-', label='LD-CPD - AUC {}'.format(round(auc_ours, 2)), linewidth=5)
    # plt.plot(thresholds_cai, means_per_threshold_cai, '.r-', label='XCC - AUC {}'.format(round(auc_cai, 2)), linewidth=5)
    # plt.plot(thresholds_kesh, means_per_threshold_kesh, '.g-', label='KMA - AUC {}'.format(round(auc_kesh, 2)), linewidth=5)
    # plt.plot(thresholds_kesh_alt, means_per_threshold_kesh_alt, '.k-', label='KM - AUC {}'.format(round(auc_kesh_alt, 2)), linewidth=5)
    plt.plot(thresholds_ours, means_per_threshold_ours, '.b-', label='LD-CPD', linewidth=5)
    plt.plot(thresholds_cai, means_per_threshold_cai, '.r-', label='XCC', linewidth=5)
    plt.plot(thresholds_kesh, means_per_threshold_kesh, '.g-', label='KMA', linewidth=5)
    plt.plot(thresholds_kesh_alt, means_per_threshold_kesh_alt, '.k-', label='KM', linewidth=5)
    plt.xlabel('FPR', fontsize=38)
    plt.ylabel('Detection Time (in 5min Intervals)', fontsize=38)
    plt.title('Average AMOC Curve for Central OK', fontsize=36)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(loc='best', fontsize=32)
    plt.ylim(0.0, max(means_per_threshold_kesh.max(), means_per_threshold_ours.max(), means_per_threshold_cai.max())+5.0)
    
    if not os.path.exists(curr_storm_save_path):
        os.mkdir(curr_storm_save_path)
    plt.savefig(os.path.join(curr_storm_save_path, 'avg_amoc{}_{}.png'.format(kesh_suffix, storm_name)))
    plt.close()


def main_mesonet_pressure(storm_name='pressure', clime=False):
    """
    In progress for MesoNet data collection

    Will be more involved. Mainly for CP location identification with storm pairings
    """
    window_size = 600 # this is fixed currently
    post_window_size = 25 # this is fixed currently
    dim = 80 # this is fixed currently
    curr_path = os.getcwd()
    save_path = os.path.join(curr_path, 'amoc_figs/')
    curr_storm_save_path = save_path+"mesonet/"+storm_name
    # cpd_path = 'results/{}_storm_feb_2024'.format(storm_name)
    # kesh_path = 'results_kesh/mesonet_feb_2024'
    # kesh_alt_path = 'results_kesh_alt/mesonet_feb_2024'
    # cai_path = 'results_cai/mesonet_feb_2024'

    cpd_path = 'results/{}_storm'.format(storm_name)

    cpd_path = 'results/{}_storm'.format(storm_name)
    kesh_path = 'results_kesh/mesonet_pressure'
    kesh_alt_path = 'results_kesh_alt/mesonet_pressure'
    if clime:
        kesh_path = 'results_kesh_clime/mesonet_pressure'
        kesh_alt_path = 'results_kesh_alt_clime/mesonet_pressure'
    cai_path = 'results_cai/mesonet_pressure'
    storm_data_path = '../data/out_{}'.format(storm_name)
    dir_files = os.listdir(cpd_path)
    storm_dir_files = os.listdir(storm_data_path)
    # storm_nums = []
    # for fi in dir_files:
    #     if 'center_storm' in fi:
    #         storm_nums.append(int(fi.split('_')[-1].split('.')[0]))
    #storm_nums = [1,3,4,5,6,7,8,9,12,13,14,15,16,17,23,24,25,26,28,32,33,34,35,38]
    storm_nums = [0, 47, 51, 103, 104, 105, 107, 127, 137, 144, 153, 162, 163, 173, 176, 200, 201, 329, 338, 384, 386, 390, 391, 392, 398, 403, 408, 409, 414, 415]

    storm_files = []
    for num in storm_nums:
        storm_files.append('{}_storm_{}_storm_data.csv'.format(storm_name, num))
    
    our_results = {}
    kesh_results = {}
    fpr_detect_pairs_ours = []
    fpr_detect_pairs_cai = []
    fpr_detect_pairs_kesh = []
    fpr_detect_pairs_kesh_alt = []
    lol_ours_fprs = []
    lol_ours_dts = []
    lol_cai_fprs = []
    lol_cai_dts = []
    lol_kesh_fprs = []
    lol_kesh_dts = []
    lol_kesh_alt_fprs = []
    lol_kesh_alt_dts = []
    unique_tuples = []
    max_detect_times = []
    storm_type_list = []
    storm_duration_list = []
    #storm_nums = [6]
    auc_ours = []
    auc_kesh = []
    auc_kesh_alt = []
    auc_cai = []
    for num in sorted(storm_nums):
        cpd_vals = np.loadtxt(os.path.join(cpd_path, "{}_storm_{}.csv".format(storm_name, num)), delimiter=',')
        cutoff = cpd_vals.shape[1]//2 # split test stats and p-vals
        pvals_cpd = cpd_vals[:, cutoff:][window_size:] # just the p-values, exclude first window size
        stats_cpd = cpd_vals[:, 0:cutoff][window_size:] # just the stats, exclude first window size

        # kesh_vals = np.loadtxt(os.path.join(kesh_path, "{}_storm_{}_final_data.csv_global_test_vals.csv".format(storm_name, num)), delimiter=',')
        # kesh_vals_alt = np.loadtxt(os.path.join(kesh_alt_path, "{}_storm_{}_final_data.csv_global_test_vals.csv".format(storm_name, num)), delimiter=',')
        # cai_vals = np.loadtxt(os.path.join(cai_path, "{}_storm_{}_final_data.csv_global_test_vals.csv".format(storm_name, num)), delimiter=',')

        kesh_vals = np.loadtxt(os.path.join(kesh_path, "{}_storm_{}_unprocessed_data.csv_global_test_vals.csv".format(storm_name, num)), delimiter=',')
        kesh_vals_alt = np.loadtxt(os.path.join(kesh_alt_path, "{}_storm_{}_unprocessed_data.csv_global_test_vals.csv".format(storm_name, num)), delimiter=',')
        cai_vals = np.loadtxt(os.path.join(cai_path, "{}_storm_{}_unprocessed_data.csv_global_test_vals.csv".format(storm_name, num)), delimiter=',')
        # print(cai_vals.shape)
        # print(kesh_vals.shape)
        # print(pvals_cpd.shape)
        # print(kesh_vals_alt.shape)
        print(num)
        actual_final_data = pd.read_csv(os.path.join(storm_data_path, '{}_storm_{}_unprocessed_data.csv').format(storm_name, num))
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
            elif (storm_end_idx - storm_start_idx) >= 600:
                continue
            else:
                unique_tuples.append(curr_tuple)
        cp_location = storm_start_idx - 1 # adjust for first differencing
        cp_location_adjustment = cp_location - window_size - 1 # adjust for first differencing
        # subtract off window_size to handle the sliding window edges at start/end of data
        first_d_time = cp_location_adjustment - post_window_size + 1 - 12
        storm_duration = storm_end_idx - storm_start_idx + 12*2
        last_d_time = cp_location_adjustment + storm_duration + 1 - post_window_size + 12 # how long is the storm contained in the window as it slides over the data - at least window_size times (right to left)

        adjustment_window_start = first_d_time - 120
        adjustment_window_end = last_d_time
        #first_d_time = 120
        #last_d_time = last_d_time - adjustment_window_start

        print("*** STORM NUMBER {} ***".format(num))
        print("Storm type {}; Duration {}".format(storm_type, storm_duration))
        print("Storm Start {} End {}".format(storm_start_idx, storm_end_idx))
        print("Adj Storm Start {} End {}".format(storm_start_idx-window_size, storm_end_idx-window_size))
        print("Final Data shape {}".format(actual_final_data.shape))
        storm_type_list.append(storm_type)
        storm_duration_list.append(storm_duration)
        max_detect_times.append(last_d_time-first_d_time)
        #print(pvals_cpd.shape, stats_cpd.shape, kesh_vals.shape, actual_final_data.shape)
        
        
        #print(stats_cpd.min(), stats_cpd.max(), pvals_cpd.max())
        pvals_cpd = pvals_cpd.min(axis=1)
        pvals_kesh = kesh_p_value(kesh_vals)
        pvals_cai = cai_p_value(cai_vals, dim=cai_vals.shape[0])

        """
        Cut out points after storm
        """
        pvals_cpd_uncut = pvals_cpd.copy()
        kesh_vals_uncut = kesh_vals.copy()
        kesh_vals_alt_uncut = kesh_vals_alt.copy()
        cai_vals_uncut = cai_vals.copy()
        pvals_cpd = pvals_cpd[0:last_d_time+1]
        pvals_kesh = pvals_kesh[0:last_d_time+1]
        pvals_cai = pvals_cai[0:last_d_time+1]
        stats_cpd = stats_cpd.max(axis=1)[0:last_d_time+1]
        kesh_vals = kesh_vals[0:last_d_time+1]
        kesh_vals_alt = kesh_vals_alt[0:last_d_time+1]
        cai_vals = cai_vals[0:last_d_time+1]

        # pvals_cpd = pvals_cpd[adjustment_window_start:adjustment_window_end+1]
        # pvals_kesh = pvals_kesh[adjustment_window_start:adjustment_window_end+1]
        # pvals_cai = pvals_cai[adjustment_window_start:adjustment_window_end+1]
        # stats_cpd = stats_cpd.max(axis=1)[adjustment_window_start:adjustment_window_end+1]
        # kesh_vals = kesh_vals[adjustment_window_start:adjustment_window_end+1]
        # kesh_vals_alt = kesh_vals_alt[adjustment_window_start:adjustment_window_end+1]
        # cai_vals = cai_vals[adjustment_window_start:adjustment_window_end+1]
        """
        """
        print(pvals_cpd.shape, kesh_vals.shape, kesh_vals_alt.shape, cai_vals.shape)
        #exit()

        # lrt_result_cpd, detect_result_cpd = amoc_p_vals(pvals_cpd, 
        #                                                 first_possible_detect_time=first_d_time,
        #                                                 last_possible_detect_time=last_d_time,
        #                                                 dim=None, # deprecated
        #                                                 thresholds=all_thresholds
        #                                                 )
        
        lrt_result_cpd, detect_result_cpd = amoc_lrt_vals(pvals_cpd, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=None, p_values=True)
        lrt_result_kesh, detect_result_kesh = amoc_lrt_vals(kesh_vals, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=None, p_values=False)
        lrt_result_kesh_alt, detect_result_kesh_alt = amoc_lrt_vals(kesh_vals_alt, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=None, p_values=False)
        lrt_result_cai, detect_result_cai = amoc_lrt_vals(cai_vals, first_possible_detect_time=first_d_time, last_possible_detect_time=last_d_time, thresholds=None, p_values=False)

        cutidx_ours = np.where(lrt_result_cpd <= 0.05)
        cutidx_kesh = np.where(lrt_result_kesh <= 0.05)
        cutidx_kesh_alt = np.where(lrt_result_kesh_alt <= 0.05)
        cutidx_cai = np.where(lrt_result_cai <= 0.05)
        auc_ours.append(auc(x=lrt_result_cpd[cutidx_ours], y=detect_result_cpd[cutidx_ours]))
        auc_kesh.append(auc(x=lrt_result_kesh[cutidx_kesh], y=detect_result_kesh[cutidx_kesh]))
        auc_kesh_alt.append(auc(x=lrt_result_kesh_alt[cutidx_kesh_alt], y=detect_result_kesh_alt[cutidx_kesh_alt]))
        auc_cai.append(auc(x=lrt_result_cai[cutidx_cai], y=detect_result_cai[cutidx_cai]))

        
        print("Detect Ours: Min {} Max {} Mean {}".format(detect_result_cpd.min(), detect_result_cpd.max(), detect_result_cpd.mean()))
        print("Detect Kesh: Min {} Max {} Mean {}".format(detect_result_kesh.min(), detect_result_kesh.max(), detect_result_kesh.mean()))
        print("Detect KeshAlt: Min {} Max {} Mean {}".format(detect_result_kesh_alt.min(), detect_result_kesh_alt.max(), detect_result_kesh_alt.mean()))
        print("Detect Cai: Min {} Max {} Mean {}".format(detect_result_cai.min(), detect_result_cai.max(), detect_result_cai.mean()))
        print("Maximum Detect Time {}".format(last_d_time-first_d_time))
        print("**************************")
        print()

        plt.plot(pvals_cpd_uncut)
        plt.axvline(cp_location_adjustment, linestyle='--', color='green')
        kesh_suffix = "_clime" if clime else ""
        plt.savefig('debugging_figs/mesonet_center/{}_{}{}.png'.format(num, "ours", kesh_suffix))
        plt.close()
        plt.plot(cai_vals_uncut)
        plt.axvline(cp_location_adjustment, linestyle='--', color='green')
        plt.savefig('debugging_figs/mesonet_center/{}_{}{}.png'.format(num, "cai", kesh_suffix))
        plt.close()
        plt.plot(kesh_vals_uncut)
        plt.axvline(cp_location_adjustment, linestyle='--', color='green')
        plt.savefig('debugging_figs/mesonet_center/{}_{}{}.png'.format(num, "kma", kesh_suffix))
        plt.close()
        plt.plot(kesh_vals_alt_uncut)
        plt.axvline(cp_location_adjustment, linestyle='--', color='green')
        plt.savefig('debugging_figs/mesonet_center/{}_{}{}.png'.format(num, "km", kesh_suffix))
        plt.close()

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
        sorted_lrt_result_kesh_alt = lrt_result_kesh_alt
        sorted_detect_result_cpd = detect_result_cpd  #sorted(detect_result_cpd, reverse=False) # 
        sorted_detect_result_cai = detect_result_cai #sorted(detect_result_cai, reverse=False)
        sorted_detect_result_kesh = detect_result_kesh  # sorted(detect_result_kesh, reverse=False) #
        sorted_detect_result_kesh_alt = detect_result_kesh_alt

        merged_list_ours = [(sorted_lrt_result_cpd[i], 
                            sorted_detect_result_cpd[i]) for i in range(0, len(lrt_result_cpd))]
        merged_list_cai = [(sorted_lrt_result_cai[i], 
                            sorted_detect_result_cai[i]) for i in range(0, len(lrt_result_cai))]
        merged_list_kesh = [(sorted_lrt_result_kesh[i], 
                            sorted_detect_result_kesh[i]) for i in range(0, len(lrt_result_kesh))]
        merged_list_kesh_alt = [(sorted_lrt_result_kesh_alt[i], 
                            sorted_detect_result_kesh_alt[i]) for i in range(0, len(lrt_result_kesh_alt))]
        
        fpr_detect_pairs_ours.append(merged_list_ours)
        fpr_detect_pairs_cai.append(merged_list_cai)
        fpr_detect_pairs_kesh.append(merged_list_kesh)
        fpr_detect_pairs_kesh_alt.append(merged_list_kesh_alt)

        lol_ours_fprs.append(sorted_lrt_result_cpd)
        lol_ours_dts.append(sorted_detect_result_cpd)
        lol_cai_fprs.append(sorted_lrt_result_cai)
        lol_cai_dts.append(sorted_detect_result_cai)
        lol_kesh_fprs.append(sorted_lrt_result_kesh)
        lol_kesh_dts.append(sorted_detect_result_kesh)
        lol_kesh_alt_fprs.append(sorted_lrt_result_kesh_alt)
        lol_kesh_alt_dts.append(sorted_detect_result_kesh_alt)

        plt.step(sorted_lrt_result_cpd, sorted_detect_result_cpd, '-.b', 
                        label='LD-CPD', where='post')
        plt.step(sorted_lrt_result_kesh, sorted_detect_result_kesh, '-.g', 
                label='KMA', where='post')
        plt.step(sorted_lrt_result_kesh_alt, sorted_detect_result_kesh_alt, '-.k', 
                label='KM', where='post')
        plt.step(sorted_lrt_result_cai, sorted_detect_result_cai, '-.r', 
                label='Cai', where='post')
        plt.ylabel("Detect Time (5min Intervals)")
        plt.xlabel("FPR")
        plt.title("{} Storm_{}".format(storm_name, num))
        plt.legend(loc='best')
        kesh_suffix = "_clime" if clime else ""
        plt.savefig(os.path.join(curr_storm_save_path, 'amoc{}_{}_{}.png'.format(kesh_suffix, storm_name, num)))
        plt.close()
    #means_per_threshold_ours, thresholds_ours, fpr_dict_ours = average_AMOC(fpr_detect_pairs_ours)
    #means_per_threshold_cai, thresholds_cai, fpr_dict_cai = average_AMOC(fpr_detect_pairs_cai)
    #means_per_threshold_kesh, thresholds_kesh, fpr_dict_kesh = average_AMOC(fpr_detect_pairs_kesh)
    means_per_threshold_ours, thresholds_ours, fpr_dict_ours = average_AMOC_new(fprs=lol_ours_fprs, dts=lol_ours_dts, debug_title='Ours_Mesonet_Pressure')
    means_per_threshold_cai, thresholds_cai, fpr_dict_cai = average_AMOC_new(fprs=lol_cai_fprs, dts=lol_cai_dts, debug_title='Cai_Mesonet_Pressure')
    means_per_threshold_kesh, thresholds_kesh, fpr_dict_kesh = average_AMOC_new(fprs=lol_kesh_fprs, dts=lol_kesh_dts, debug_title='KMA_Mesonet_Pressure')
    means_per_threshold_kesh_alt, thresholds_kesh_alt, fpr_dict_kesh_alt = average_AMOC_new(fprs=lol_kesh_alt_fprs, dts=lol_kesh_alt_dts, debug_title='KM_Mesonet_Pressure')

    sns.boxplot(auc_ours)
    plt.title('{} Storm AUC Ours FPR <= 0.05'.format(storm_name))
    plt.ylabel('Detection Time')
    plt.savefig(os.path.join(curr_storm_save_path, 'auc_ours_{}.png'.format(storm_name)))
    plt.close()
    sns.boxplot(auc_kesh)
    plt.title('{} Storm AUC KMA FPR <= 0.05'.format(storm_name))
    plt.ylabel('Detection Time')
    kesh_suffix = "_clime" if clime else ""
    plt.savefig(os.path.join(curr_storm_save_path, 'auc_kesh{}_{}.png'.format(kesh_suffix, storm_name)))
    plt.close()
    sns.boxplot(auc_kesh_alt)
    plt.title('{} Storm AUC KM FPR <= 0.05'.format(storm_name))
    plt.ylabel('Detection Time')
    plt.savefig(os.path.join(curr_storm_save_path, 'auc_kesh_alt{}_{}.png'.format(kesh_suffix, storm_name)))
    plt.close()
    sns.boxplot(auc_cai)
    plt.title('{} Storm AUC XCC FPR <= 0.05'.format(storm_name))
    plt.ylabel('Detection Time')
    plt.savefig(os.path.join(curr_storm_save_path, 'auc_cai_{}.png'.format(storm_name)))
    plt.close()

    plt.xticks([1, 2, 3, 4], ["LD-CPD", "XCC", "KMA", "KM"])
    plt.xticks(rotation=45, ha='right')
    plot_confidence_interval(x=1, values=auc_ours, color='blue', horizontal_line_width=0.25)
    plot_confidence_interval(x=2, values=auc_cai, color='red', horizontal_line_width=0.25)
    plot_confidence_interval(x=3, values=auc_kesh, color='green', horizontal_line_width=0.25)
    plot_confidence_interval(x=4, values=auc_kesh_alt, color='black', horizontal_line_width=0.25)
    #plt.ylim(-10.0, 40.0)
    plt.ylim(0.0)
    plt.title("95% Confidence Intervals AMOC-AUC {}".format("Pressure Sensors"))
    plt.ylabel("AMOC-AUC")
    #plt.xlabel("FPR")
    plt.tight_layout()
    plt.savefig(os.path.join(curr_storm_save_path, 'auc_conf{}_{}.png'.format(kesh_suffix, storm_name)))
    #plt.savefig(os.path.join(curr_dim_path+'/avg_amoc/', 'auc_conf_{}_dim_{}.png'.format(sim_type, curr_dim)))
    plt.close()
    np.savetxt(os.path.join(curr_storm_save_path, 'auc_ours_{}.csv'.format(storm_name)), auc_ours, delimiter=',')
    np.savetxt(os.path.join(curr_storm_save_path, 'auc_cai_{}.csv'.format(storm_name)), auc_cai, delimiter=',')
    np.savetxt(os.path.join(curr_storm_save_path, 'auc_kesh{}_{}.csv'.format(kesh_suffix, storm_name)), auc_kesh, delimiter=',')
    np.savetxt(os.path.join(curr_storm_save_path, 'auc_kesh_alt{}_{}.csv'.format(kesh_suffix, storm_name)), auc_kesh_alt, delimiter=',')
    auc_ours = np.array(auc_ours).mean()
    auc_kesh = np.array(auc_kesh).mean()
    auc_kesh_alt = np.array(auc_kesh_alt).mean()
    auc_cai = np.array(auc_cai).mean()

    # auc_ours = np.array(auc_ours).mean()
    # auc_kesh = np.array(auc_kesh).mean()
    # auc_kesh_alt = np.array(auc_kesh_alt).mean()
    # auc_cai = np.array(auc_cai).mean()

    sns.boxplot(storm_duration_list)
    plt.savefig('debugging_figs/pressure_duration_plot.png')
    plt.ylabel('Duration (5min Intervals)')
    plt.title("Storm Durations")
    plt.close()

    
    # plot_idx_counter = 1
    # ylim_max = 0.0
    # for fpr_r in [0.01, 0.05, 0.1]:
    #     plt.xticks([1, 2, 3], ["{}_ours".format(fpr_r), "{}_cai".format(fpr_r), "{}_kesh".format(fpr_r)])
    #     #plt.xticks([1, 2], ["{}_ours".format(fpr_r), "{}_kesh".format(fpr_r)])
    #     plt.xticks(rotation=45, ha='right')
    #     our_d_vals = fpr_dict_ours[fpr_r]
    #     cai_d_vals = fpr_dict_cai[fpr_r]
    #     kesh_d_vals = fpr_dict_kesh[fpr_r]
    #     our_max = max(our_d_vals)
    #     cai_max = max(cai_d_vals)
    #     kesh_max = max(kesh_d_vals)
    #     glob_max = max([our_max, cai_max, kesh_max])
    #     #glob_max = max([our_max, kesh_max])
    #     if glob_max > ylim_max:
    #         ylim_max = glob_max
    #     plot_confidence_interval(x=1, values=our_d_vals, z=1.96, color='#2187bb', horizontal_line_width=0.25)
    #     plot_idx_counter += 1
    #     plot_confidence_interval(x=2, values=cai_d_vals, z=1.96, color='#2187bb', horizontal_line_width=0.25)
    #     plot_idx_counter += 1
    #     plot_confidence_interval(x=3, values=kesh_d_vals, z=1.96, color='#2187bb', horizontal_line_width=0.25)
    #     plot_idx_counter += 1
    #     #plot_confidence_interval(x=2, values=kesh_d_vals, z=1.96, color='#2187bb', horizontal_line_width=0.25)
    #     plt.ylim(0.0, ylim_max)
    #     plt.title("Confidence Intervals {} Storm".format(storm_name))
    #     plt.ylabel("Detect Time (5min Intervals)")
    #     plt.xlabel("FPR")
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(curr_storm_save_path, 'conf_avg_amoc_{}_{}.png'.format(storm_name, fpr_r)))
    #     plt.close()

    threshold_idx_mask = np.where(thresholds_ours <= 0.05) # THESE ARE IDENTICAL INDICES ACROSS THE ALGORITHMS
    means_per_threshold_ours = means_per_threshold_ours[threshold_idx_mask]
    means_per_threshold_cai = means_per_threshold_cai[threshold_idx_mask]
    means_per_threshold_kesh = means_per_threshold_kesh[threshold_idx_mask]
    means_per_threshold_kesh_alt = means_per_threshold_kesh_alt[threshold_idx_mask]
    thresholds_ours = thresholds_ours[threshold_idx_mask]
    thresholds_cai = thresholds_cai[threshold_idx_mask]
    thresholds_kesh = thresholds_kesh[threshold_idx_mask]
    thresholds_kesh_alt = thresholds_kesh_alt[threshold_idx_mask]

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
    plt.figure(figsize=(25, 14))
    #auc_ours = auc(x=thresholds_ours, y=means_per_threshold_ours)
    #auc_cai = auc(x=thresholds_cai, y=means_per_threshold_cai)
    #auc_kesh = auc(x=thresholds_kesh, y=means_per_threshold_kesh)
    #auc_kesh_alt = auc(x=thresholds_kesh_alt, y=means_per_threshold_kesh_alt)
    plt.plot(thresholds_ours, means_per_threshold_ours, '.b-', label='LD-CPD', linewidth=5)
    plt.plot(thresholds_cai, means_per_threshold_cai, '.r-', label='XCC', linewidth=5)
    plt.plot(thresholds_kesh, means_per_threshold_kesh, '.g-', label='KMA', linewidth=5)
    plt.plot(thresholds_kesh_alt, means_per_threshold_kesh_alt, '.k-', label='KM', linewidth=5)
    # plt.plot(thresholds_ours, means_per_threshold_ours, '.b-', label='LD-CPD - AUC {}'.format(round(auc_ours, 2)), linewidth=5)
    # plt.plot(thresholds_cai, means_per_threshold_cai, '.r-', label='XCC - AUC {}'.format(round(auc_cai, 2)), linewidth=5)
    # plt.plot(thresholds_kesh, means_per_threshold_kesh, '.g-', label='KMA - AUC {}'.format(round(auc_kesh, 2)), linewidth=5)
    # plt.plot(thresholds_kesh_alt, means_per_threshold_kesh_alt, '.k-', label='KM - AUC {}'.format(round(auc_kesh_alt, 2)), linewidth=5)
    plt.xlabel('FPR', fontsize=38)
    plt.ylabel('Detection Time (in 5min Intervals)', fontsize=38)
    plt.title('Average AMOC Curve for OK (Pressure Sensors)', fontsize=36)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(loc='best', fontsize=32)
    plt.ylim(0.0, max(means_per_threshold_kesh.max(), means_per_threshold_ours.max(), means_per_threshold_cai.max())+5.0)
    
    if not os.path.exists(curr_storm_save_path):
        os.mkdir(curr_storm_save_path)
    plt.savefig(os.path.join(curr_storm_save_path, 'avg_amoc{}_{}.png'.format(kesh_suffix, storm_name)))
    plt.close()


        


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sims', action='store_true')
    parser.add_argument('--mesonet', action='store_true')
    parser.add_argument('--avesenov', action='store_true')
    parser.add_argument('--pressure', action='store_true')
    parser.add_argument('--clime', action='store_true')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_args()
    if args.sims:
        main_sims(clime=args.clime)
    if args.mesonet:
        main_mesonet(clime=args.clime)
    if args.avesenov:
        main_sims_avesnov()
    if args.pressure:
        main_mesonet_pressure(clime=args.clime)
