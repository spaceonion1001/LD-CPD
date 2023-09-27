import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import chi2, multivariate_normal, zscore
from tqdm.notebook import tqdm
from scipy.stats import norm
from numba import jit
import argparse
sns.set()
#sns.set_style('dark')
sns.set_palette("viridis")


def kesh_p_value(test_statistics):
    """
    Calculate z-score, then take survival function of normal dist
    """
    
    z_scores = zscore(test_statistics)
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


# OURS
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
    for k in tqdm(range(len(thresholds))):
        threshold = thresholds[k]
        summed_days = 0
        
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
        
        mean_days = summed_days / len(AMOC_points)
        mean_days_per_threshold.append(mean_days)
    
    return mean_days_per_threshold, thresholds

def amoc_lrt_vals(lrt_vals, first_possible_detect_time, last_possible_detect_time, use_p_vals=False, thresholds=None):
    # input should be padded vals
    p_vals = np.array([chi2.pdf(lrt_vals[:, i], 2) for i in range(lrt_vals.shape[1])]).T
    p_vals_corrected = apply_fdr_correction(p_vals)
    fprs = []
    detections = []
    fpr_array, detection_array = amoc_gen(lrt_vals, # use LRT vals here
                                          first_detect_time=first_possible_detect_time, 
                                          last_detect_time=last_possible_detect_time, 
                                          max_time_detection=last_possible_detect_time-first_possible_detect_time,
                                          use_p_vals=use_p_vals,
                                          thresholds=thresholds)
    #fprs.append(fpr_array)
    #detections.append(detection_array)
    fpr_result = np.array(fpr_array)
    detect_result = np.array(detection_array)
    #fpr_result = fprs.max(axis=0)
    #detect_result = detections.min(axis=0)
    
    return fpr_result, detect_result
    
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

@jit(nopython=True)
def amoc_gen_cai(alarms, first_detect_time, last_detect_time, max_time_detection, p, thresholds=None):
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
            if p_value >= thr:
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

def amoc_lrt_vals_cai(lrt_vals, dim, first_possible_detect_time, last_possible_detect_time):
    # input should be padded vals
    #p_vals = np.array([chi2.pdf(lrt_vals[:, i], 2) for i in range(lrt_vals.shape[1])]).T
    fprs = []
    detections = []
    
    fprs, detections = amoc_gen_cai(lrt_vals, 
                                    first_detect_time=first_possible_detect_time, 
                                    last_detect_time=last_possible_detect_time, 
                                    max_time_detection=last_possible_detect_time-first_possible_detect_time,
                                    p=dim)

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
    seeds = np.arange(50, 60)
    #seeds = np.arange(50, 54)
    sim_types = ['anderson_residual_block', 'anderson_residual_unstructured']
    #sim_types = ['orthogonal']
    #sim_types = ['cai_model_one']
    dims = [20, 40, 60, 80]
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
            statset_ours = []
            statset_cai = []
            statset_kesh = []
            for curr_seed in seeds:
                cpd_path='./simulation_results/{}_{}/{}/lrt_vals.csv'.format(sim_type, curr_dim, curr_seed)
                xia_path='./simulation_results_cai/{}_{}/{}/global_test_vals.csv'.format(sim_type, curr_dim, curr_seed)
                kesh_path='./simulation_results_kesh/{}_{}/{}/global_test_vals.csv'.format(sim_type, curr_dim, curr_seed)
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
                cp_location = 200
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
                
                all_thresholds = sorted(list(set(np.concatenate((pvals_cpd, pvals_cai, pvals_kesh)))))
                
                
                lrt_result_cpd, detect_result_cpd = amoc_p_vals(pvals_cpd, 
                                                                first_possible_detect_time=first_d_time,
                                                                last_possible_detect_time=last_d_time,
                                                                dim=curr_dim,
                                                                thresholds=all_thresholds
                                                                )
                lrt_result_cai, detect_result_cai = amoc_p_vals(pvals_cai, 
                                                                first_possible_detect_time=first_d_time,
                                                                last_possible_detect_time=last_d_time,
                                                                dim=curr_dim,
                                                                thresholds=all_thresholds
                                                                )
                lrt_result_kesh, detect_result_kesh = amoc_p_vals(pvals_kesh, 
                                                                first_possible_detect_time=first_d_time,
                                                                last_possible_detect_time=last_d_time,
                                                                dim=curr_dim,
                                                                thresholds=all_thresholds
                                                                )
                
                our_seed_dict[curr_seed] = (lrt_result_cpd, detect_result_cpd)
                cai_seed_dict[curr_seed] = (lrt_result_cai, detect_result_cai)
                kesh_seed_dict[curr_seed] = (lrt_result_kesh, detect_result_kesh)
                our_results[sim_type] = our_seed_dict
                cai_results[sim_type] = cai_seed_dict
                kesh_results[sim_type] = kesh_seed_dict
                print(pvals_cpd.shape, stats_cpd.shape, xia_cpd.shape, kesh_cpd.shape)

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

                statset_ours.append(stats_cpd.flatten())
                statset_cai.append(xia_cpd.flatten())
                statset_kesh.append(kesh_cpd.flatten())
            
            means_per_threshold_ours, thresholds_ours = average_AMOC(fpr_detect_pairs_ours)
            means_per_threshold_cai, thresholds_cai = average_AMOC(fpr_detect_pairs_cai)
            means_per_threshold_kesh, thresholds_kesh = average_AMOC(fpr_detect_pairs_kesh)
            plt.plot(thresholds_ours, means_per_threshold_ours, '.b-', label='Ours')
            plt.plot(thresholds_cai, means_per_threshold_cai, '.r-', label='Cai')
            plt.plot(thresholds_kesh, means_per_threshold_kesh, '.g-', label='Kesh')
            plt.xlabel('FPR')
            plt.ylabel('Detection Time')
            plt.title('Average AMOC Curve for {}'.format(sim_type))
            plt.legend(loc='best')
            plt.savefig(os.path.join(curr_dim_path, 'avg_amoc_{}_dim_{}.png'.format(sim_type, curr_dim)))
            plt.close()

def main_mesonet():
    """
    In progress for MesoNet data collection

    Will be more involved. Mainly for CP location identification with storm pairings
    """
    pass


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--sims', actions='store_true')
    parser.add_argument('--mesonet', action='store_true')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main_sims()
