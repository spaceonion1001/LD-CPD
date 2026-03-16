import numpy as np
from scipy.stats import wilcoxon, levene
import scipy.stats as st
import os
import statistics
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import pandas as pd
import argparse

matplotlib.use('Agg')
sns.set()


def f_test(x, y, alt="two_sided"):
    """
    Calculates the F-test.
    :param x: The first group of data
    :param y: The second group of data
    :param alt: The alternative hypothesis, one of "two_sided" (default), "greater" or "less"
    :return: a tuple with the F statistic value and the p-value.
    """
    df1 = len(x) - 1
    df2 = len(y) - 1
    f = x.var() / y.var()
    if alt == "greater":
        p = 1.0 - st.f.cdf(f, df1, df2)
    elif alt == "less":
        p = st.f.cdf(f, df1, df2)
    else:
        # two-sided by default
        # Crawley, the R book, p.355
        p = 2.0*(1.0 - st.f.cdf(f, df1, df2))
    return f, p

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
    plt.plot([x, x], [top, bottom], color=color, linewidth=3)
    plt.plot([left, right], [top, top], color=color, linewidth=3)
    plt.plot([left, right], [bottom, bottom], color=color, linewidth=3)
    plt.plot(x, mean, 'o', color='#f44336', markersize=10)
    #plt.plot(x, median, 'o', color='yellow')
    return mean, confidence_interval

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clime', action='store_true')
    return parser.parse_args()

def sim_display_name(sim_type):
    if sim_type == 'orthogonal_small':
        return 'Single'
    if sim_type == 'orthogonal_cross_block':
        return 'Multiple'
    if sim_type == 'orthogonal_multiple_block':
        return 'Indiv Coeffs'
    if sim_type == 'orthogonal_hard':
        return 'Indiv Coeffs Subsets'
    if sim_type == 'orthogonal_cross_hard':
        return 'Multiple Subsets'
    if sim_type == 'cai_model_one':
        return 'Banded Matrix Change'
    if sim_type == 'cai_model_three':
        return 'Scattered Matrix Change'
    return sim_type

def summarize_one_sided(sim_label, dim_label, data_map):
    means = {k: float(np.mean(v)) for k, v in data_map.items()}
    best = min(means, key=means.get)
    comparisons = []
    for other, arr in data_map.items():
        if other == best:
            continue
        stat, pval = wilcoxon(data_map[best], arr, alternative='less', zero_method='wilcox', correction=False)
        comparisons.append({
            "sim": sim_label,
            "dim": dim_label,
            "best": best,
            "other": other,
            "best_mean": means[best],
            "other_mean": means[other],
            "statistic": float(stat),
            "p_value": float(pval),
        })
    return means, best, comparisons

if __name__ == '__main__':
    args = get_args()
    kesh_suffix = "_clime" if args.clime else ""
    base_dir = '/home/dink/Documents/Research/Correlation-Changepoint-Detection/amoc_figs'
    sim_rows = []
    mesonet_rows = []
    for sim in ['orthogonal_small', 'orthogonal_cross_block', 'orthogonal_multiple_block', 'orthogonal_hard', 'cai_model_one']:
    # for sim in ['cai_model_one', 'cai_model_three']:
        for dim in [20, 40, 60, 80]:
            block_ours = np.loadtxt(os.path.join(base_dir, '{}/avg_amoc/auc_{}_{}_dim_{}.csv'.format(dim, 'ours', sim, dim)), delimiter=',')
            block_cai = np.loadtxt(os.path.join(base_dir, '{}/avg_amoc/auc_{}_{}_dim_{}.csv'.format(dim, 'cai', sim, dim)), delimiter=',')
            block_kesh = np.loadtxt(os.path.join(base_dir, '{}/avg_amoc/auc_kesh{}_{}_dim_{}.csv'.format(dim, kesh_suffix, sim, dim)), delimiter=',')
            block_kesh_alt = np.loadtxt(os.path.join(base_dir, '{}/avg_amoc/auc_kesh_alt{}_{}_dim_{}.csv'.format(dim, kesh_suffix, sim, dim)), delimiter=',')
            print("********************")
            print(sim, dim)
            means, best, comparisons = summarize_one_sided(
                sim_label=sim,
                dim_label=dim,
                data_map={
                    "LD-CPD": block_ours,
                    "XCC": block_cai,
                    "KMA": block_kesh,
                    "KM": block_kesh_alt,
                },
            )
            sim_rows.append({
                "sim": sim,
                "dim": dim,
                "means": means,
                "best": best,
                "wilcoxon": comparisons,
                "variance": None,  # filled below
            })
            #print(block_ours)
            if block_ours.mean() <= block_cai.mean() and block_ours.mean() <= block_kesh.mean() and block_ours.mean() <= block_kesh_alt.mean():
                print("Ours {} XCC {} KMA {} KM {}".format(block_ours.mean(), block_cai.mean(), block_kesh.mean(), block_kesh_alt.mean()))
                print("vs Cai", wilcoxon(block_ours, block_cai, alternative='less', zero_method='wilcox', correction=False))
                print("vs KMA", wilcoxon(block_ours, block_kesh, alternative='less', zero_method='wilcox', correction=False))
                print("vs KM", wilcoxon(block_ours, block_kesh_alt, alternative='less', zero_method='wilcox', correction=False))
            elif block_cai.mean() < block_ours.mean() and block_cai.mean() <= block_kesh.mean() and block_cai.mean() <= block_kesh_alt.mean():
                print("Ours {} XCC {} KMA {} KM {}".format(block_ours.mean(), block_cai.mean(), block_kesh.mean(), block_kesh_alt.mean()))
                print("Cai vs Ours", wilcoxon(block_cai, block_ours, alternative='less', zero_method='wilcox', correction=False))
                print("Cai vs KMA", wilcoxon(block_cai, block_kesh, alternative='less', zero_method='wilcox', correction=False))
                print("Cai vs KM", wilcoxon(block_cai, block_kesh_alt, alternative='less', zero_method='wilcox', correction=False))
            elif block_kesh.mean() < block_ours.mean() and block_kesh.mean() <= block_cai.mean() and block_kesh.mean() <= block_kesh_alt.mean():
                print("Ours {} XCC {} KMA {} KM {}".format(block_ours.mean(), block_cai.mean(), block_kesh.mean(), block_kesh_alt.mean()))
                print("KMA vs Ours", wilcoxon(block_kesh, block_ours, alternative='less', zero_method='wilcox', correction=False))
                print("KMA vs Cai", wilcoxon(block_kesh, block_cai, alternative='less', zero_method='wilcox', correction=False))
                print("KMA vs KM", wilcoxon(block_kesh, block_kesh_alt, alternative='less', zero_method='wilcox', correction=False))
            else:
                print("Ours {} XCC {} KMA {} KM {}".format(block_ours.mean(), block_cai.mean(), block_kesh.mean(), block_kesh_alt.mean()))
                print("KM vs Ours", wilcoxon(block_kesh_alt, block_ours, alternative='less', zero_method='wilcox', correction=False))
                print("KM vs Cai", wilcoxon(block_kesh_alt, block_cai, alternative='less', zero_method='wilcox', correction=False))
                print("KM vs KMA", wilcoxon(block_kesh_alt, block_kesh, alternative='less', zero_method='wilcox', correction=False))
            plt.figure(figsize=(12, 9))

            plt.xticks([1, 2, 3, 4], ["LD-CPD", "XCC", "KMA", "KM"], fontsize=22)
            plt.xticks(rotation=45, ha='right')

            plot_confidence_interval(x=1, values=block_ours, color='blue', horizontal_line_width=0.25, constrain=True)
            plot_confidence_interval(x=2, values=block_cai, color='red', horizontal_line_width=0.25, constrain=True)
            plot_confidence_interval(x=3, values=block_kesh, color='green', horizontal_line_width=0.25, constrain=True)
            plot_confidence_interval(x=4, values=block_kesh_alt, color='black', horizontal_line_width=0.25, constrain=True)
            #plt.ylim(-10.0, 40.0)
            plt.ylim(0.0)
            if sim == 'orthogonal_small_block':
                plt.title("95% Confidence Intervals TR-AMOC-AUC {} Dim {}".format("Single", dim), fontsize=22)
            elif sim == 'orthogonal_cross_block':
                plt.title("95% Confidence Intervals TR-AMOC-AUC {} Dim {}".format("Multiple", dim), fontsize=22)
            elif sim == 'orthogonal_multiple_block':
                plt.title("95% Confidence Intervals TR-AMOC-AUC {} Dim {}".format("Indiv Coeffs", dim), fontsize=22)
            elif sim == 'orthogonal_hard':
                plt.title("95% Confidence Intervals TR-AMOC-AUC {} Dim {}".format("Indiv Coeffs Subsets", dim), fontsize=22)
            elif sim == 'orthogonal_cross_hard':
                plt.title("95% Confidence Intervals TR-AMOC-AUC {} Dim {}".format("Multiple Subsets", dim), fontsize=22)
            elif sim == 'cai_model_one':
                plt.title("95% Confidence Intervals TR-AMOC-AUC {} Dim {}".format("Banded Matrix Change", dim), fontsize=22)
            elif sim == 'cai_model_three':
                plt.title("95% Confidence Intervals TR-AMOC-AUC {} Dim {}".format("Scattered Matrix Change", dim), fontsize=22)
            plt.ylabel("TR-AMOC-AUC", fontsize=26)
            #plt.xlabel("FPR", fontsize=26)
            plt.tight_layout()
            plt.savefig('lrt_test_figs/auc_conf_{}_dim_{}{}.png'.format(sim, dim, kesh_suffix))
            plt.close()
            # print("Levene All {}".format(levene(block_ours, block_cai, block_kesh, block_kesh_alt, center='mean')))
            # print("Levene XCC {}".format(levene(block_ours, block_cai, center='mean')))
            # print("Levene KMA {}".format(levene(block_ours, block_kesh, center='mean')))
            # print("Levene KM {}".format(levene(block_ours, block_kesh_alt, center='mean')))
            print(">>>>><><><><><><<<<<<<")
            print("VARIANCE TESTS")
            if block_ours.mean() <= block_cai.mean() and block_ours.mean() <= block_kesh.mean() and block_ours.mean() <= block_kesh_alt.mean():
                x = block_ours
                variance_tests = [
                    ("LD-CPD", "XCC", f_test(x, y=block_cai, alt="less")),
                    ("LD-CPD", "KMA", f_test(x, y=block_kesh, alt="less")),
                    ("LD-CPD", "KM", f_test(x, y=block_kesh_alt, alt="less")),
                ]
            elif block_cai.mean() < block_ours.mean() and block_cai.mean() <= block_kesh.mean() and block_cai.mean() <= block_kesh_alt.mean():
                x = block_cai
                variance_tests = [
                    ("XCC", "LD-CPD", f_test(x, y=block_ours, alt="less")),
                    ("XCC", "KMA", f_test(x, y=block_kesh, alt="less")),
                    ("XCC", "KM", f_test(x, y=block_kesh_alt, alt="less")),
                ]
            elif block_kesh.mean() < block_ours.mean() and block_kesh.mean() <= block_cai.mean() and block_kesh.mean() <= block_kesh_alt.mean():
                x = block_kesh
                variance_tests = [
                    ("KMA", "LD-CPD", f_test(x, y=block_ours, alt="less")),
                    ("KMA", "XCC", f_test(x, y=block_cai, alt="less")),
                    ("KMA", "KM", f_test(x, y=block_kesh_alt, alt="less")),
                ]
            else:
                x = block_kesh_alt
                variance_tests = [
                    ("KM", "LD-CPD", f_test(x, y=block_ours, alt="less")),
                    ("KM", "XCC", f_test(x, y=block_cai, alt="less")),
                    ("KM", "KMA", f_test(x, y=block_kesh, alt="less")),
                ]
            sim_rows[-1]["variance"] = variance_tests
            print(">>>>><><><><><><<<<<<<")

            df = pd.DataFrame()
            df['LD-CPD'] = block_ours
            df['XCC'] = block_cai
            df['KMA'] = block_kesh
            df['KM'] = block_kesh_alt
            plt.figure(figsize=(12, 9))
            my_pal = {"LD-CPD": "b", "XCC": "red", "KMA":"green", "KM":"grey"}
            b = sns.boxplot(data=df, palette=my_pal)
            if sim == 'orthogonal_small':
                plt.title('TR-AMOC-AUC Distribution {} Dim {}'.format("Single", dim), fontsize=24)
            elif sim == 'orthogonal_cross_block':
                plt.title('TR-AMOC-AUC Distribution {} Dim {}'.format("Multiple", dim), fontsize=24)
            elif sim == 'orthogonal_multiple_block':
                plt.title('TR-AMOC-AUC Distribution {} Dim {}'.format("Individual Coeffs", dim), fontsize=24)
            elif sim == 'orthogonal_cross_hard':
                plt.title('TR-AMOC-AUC Distribution {} Dim {}'.format("Multiple Subsets", dim), fontsize=24)
            elif sim == 'orthogonal_hard':
                plt.title('TR-AMOC-AUC Distribution {} Dim {}'.format("Indiv Coeffs Subsets", dim), fontsize=24)
            plt.ylabel("TR-AMOC-AUC", fontsize=26)
            _, xlabels = plt.xticks()
            b.set_xticklabels(xlabels, size = 26)
            plt.savefig('lrt_test_figs/auc_boxplot_{}_dim_{}{}.png'.format(sim, dim, kesh_suffix))
            plt.close()
            # print("Levene {}".format(levene(block_ours, block_cai, block_kesh, block_kesh_alt, center='mean')))
            # print("Levene {}".format(levene(block_ours, block_cai, block_kesh, block_kesh_alt, center='mean')))
            # print("Levene {}".format(levene(block_ours, block_cai, block_kesh, block_kesh_alt, center='mean')))
            # print("Levene {}".format(levene(block_ours, block_cai, block_kesh, block_kesh_alt, center='mean')))
            print("********************")
            print()
    for storm in ['center', 'pressure']:
        block_ours = np.loadtxt(os.path.join(base_dir+'/mesonet/', '{}/auc_{}_{}.csv'.format(storm, 'ours', storm)), delimiter=',')
        block_cai = np.loadtxt(os.path.join(base_dir+'/mesonet/', '{}/auc_{}_{}.csv'.format(storm, 'cai', storm)), delimiter=',')
        block_kesh = np.loadtxt(os.path.join(base_dir+'/mesonet', '{}/auc_kesh{}_{}.csv'.format(storm, kesh_suffix, storm)), delimiter=',')
        block_kesh_alt = np.loadtxt(os.path.join(base_dir+'/mesonet/', '{}/auc_kesh_alt{}_{}.csv'.format(storm, kesh_suffix, storm)), delimiter=',')
        print("********************")
        print(storm)
        means, best, comparisons = summarize_one_sided(
            sim_label=storm,
            dim_label="mesonet",
            data_map={
                "LD-CPD": block_ours,
                "XCC": block_cai,
                "KMA": block_kesh,
                "KM": block_kesh_alt,
            },
        )
        mesonet_rows.append({
            "sim": storm,
            "dim": "mesonet",
            "means": means,
            "best": best,
            "wilcoxon": comparisons,
            "variance": None,  # filled below
        })
        #print(block_ours)
        print("vs Cai", wilcoxon(block_ours, block_cai, alternative='less', zero_method='wilcox', correction=False))
        print("vs KMA", wilcoxon(block_ours, block_kesh, alternative='less', zero_method='wilcox', correction=False))
        print("vs KM", wilcoxon(block_ours, block_kesh_alt, alternative='less', zero_method='wilcox', correction=False))
        plt.figure(figsize=(12, 9))

        plt.xticks([1, 2, 3, 4], ["LD-CPD", "XCC", "KMA", "KM"], fontsize=22)
        plt.xticks(rotation=45, ha='right')

        plot_confidence_interval(x=1, values=block_ours, color='blue', horizontal_line_width=0.25, constrain=True)
        plot_confidence_interval(x=2, values=block_cai, color='red', horizontal_line_width=0.25, constrain=True)
        plot_confidence_interval(x=3, values=block_kesh, color='green', horizontal_line_width=0.25, constrain=True)
        plot_confidence_interval(x=4, values=block_kesh_alt, color='black', horizontal_line_width=0.25, constrain=True)
        #plt.ylim(-10.0, 40.0)
        plt.ylim(0.0)
        if storm == 'center':
            plt.title("95% Confidence Intervals TR-AMOC-AUC {}".format("Center"), fontsize=22)
        elif storm == 'pressure':
            plt.title("95% Confidence Intervals TR-AMOC-AUC {}".format("Pressure"), fontsize=22)
        plt.ylabel("TR-AMOC-AUC", fontsize=26)
        #plt.xlabel("FPR", fontsize=26)
        plt.tight_layout()
        plt.savefig('lrt_test_figs/auc_conf_{}{}.png'.format(storm, kesh_suffix))
        plt.close()
        # print("Levene {}".format(levene(block_ours, block_cai, block_kesh, block_kesh_alt, center='mean')))
        # print("Levene {}".format(levene(block_ours, block_cai, center='mean')))
        # print("Levene {}".format(levene(block_ours, block_kesh, center='mean')))
        # print("Levene {}".format(levene(block_ours, block_kesh_alt, center='mean')))
        x = block_ours
        variance_tests = [
            ("LD-CPD", "XCC", f_test(x, y=block_cai, alt="less")),
            ("LD-CPD", "KMA", f_test(x, y=block_kesh, alt="less")),
            ("LD-CPD", "KM", f_test(x, y=block_kesh_alt, alt="less")),
        ]
        mesonet_rows[-1]["variance"] = variance_tests

        df = pd.DataFrame()
        df['LD-CPD'] = block_ours
        df['XCC'] = block_cai
        df['KMA'] = block_kesh
        df['KM'] = block_kesh_alt
        plt.figure(figsize=(12, 9))
        my_pal = {"LD-CPD": "b", "XCC": "red", "KMA":"green", "KM":"grey"}
        b = sns.boxplot(data=df, palette=my_pal)
        if storm == 'center':
            plt.title('TR-AMOC-AUC Distribution {}'.format("Center"), fontsize=24)
        elif storm == 'pressure':
            plt.title('TR-AMOC-AUC Distribution {}'.format("Pressure"), fontsize=24)
        plt.ylabel("TR-AMOC-AUC", fontsize=26)
        _, xlabels = plt.xticks()
        b.set_xticklabels(xlabels, size = 26)
        plt.savefig('lrt_test_figs/auc_boxplot_{}{}.png'.format(storm, kesh_suffix))
        plt.close()
        print("********************")
        print()

    # Save human-readable summaries
    sim_out_dir = '/home/dink/Documents/Research/Correlation-Changepoint-Detection/results/simulation_results'
    os.makedirs(sim_out_dir, exist_ok=True)
    sim_txt_path = os.path.join(sim_out_dir, 'wilcoxon_one_sided{}.txt'.format(kesh_suffix))
    with open(sim_txt_path, 'w') as f:
        for row in sim_rows:
            f.write("SIM {} DIM {}\n".format(sim_display_name(row["sim"]), row["dim"]))
            f.write("  Means:\n")
            for k, v in row["means"].items():
                f.write("    {}: {:.6f}\n".format(k, v))
            f.write("  Wilcoxon (one-sided, best < other):\n")
            for comp in row["wilcoxon"]:
                f.write(
                    "    {} vs {}: W={:.6f} p={:.6g} (means {:.6f} < {:.6f})\n".format(
                        comp["best"], comp["other"], comp["statistic"], comp["p_value"],
                        comp["best_mean"], comp["other_mean"]
                    )
                )
            f.write("  Variance (F-test, one-sided, best < other):\n")
            for left, right, res in row["variance"]:
                f.write("    {} vs {}: F={:.6f} p={:.6g}\n".format(left, right, res[0], res[1]))
            f.write("\n")

    mesonet_out_dir = '/home/dink/Documents/Research/Correlation-Changepoint-Detection/amoc_figs/mesonet'
    os.makedirs(mesonet_out_dir, exist_ok=True)
    mesonet_txt_path = os.path.join(mesonet_out_dir, 'wilcoxon_one_sided{}.txt'.format(kesh_suffix))
    with open(mesonet_txt_path, 'w') as f:
        for row in mesonet_rows:
            f.write("STORM {}\n".format(row["sim"]))
            f.write("  Means:\n")
            for k, v in row["means"].items():
                f.write("    {}: {:.6f}\n".format(k, v))
            f.write("  Wilcoxon (one-sided, best < other):\n")
            for comp in row["wilcoxon"]:
                f.write(
                    "    {} vs {}: W={:.6f} p={:.6g} (means {:.6f} < {:.6f})\n".format(
                        comp["best"], comp["other"], comp["statistic"], comp["p_value"],
                        comp["best_mean"], comp["other_mean"]
                    )
                )
            f.write("  Variance (F-test, one-sided, best < other):\n")
            for left, right, res in row["variance"]:
                f.write("    {} vs {}: F={:.6f} p={:.6g}\n".format(left, right, res[0], res[1]))
            f.write("\n")
    # block_ours = np.loadtxt('debugging_figs/detect_times_0.05_Ours_Block_20.csv', delimiter=',')
    # block_cai = np.loadtxt('debugging_figs/detect_times_0.05_Cai_Block_20.csv', delimiter=',')
    # block_KM = np.loadtxt('debugging_figs/detect_times_0.05_KM_Block_20.csv', delimiter=',')
    # block_KMA = np.loadtxt('debugging_figs/detect_times_0.05_KMA_Block_20.csv', delimiter=',')

    
    # banded_ours = np.loadtxt('debugging_figs/detect_times_0.05_Ours_Banded_20.csv', delimiter=',')
    # banded_cai = np.loadtxt('debugging_figs/detect_times_0.05_Cai_Banded_20.csv', delimiter=',')
    # banded_KM = np.loadtxt('debugging_figs/detect_times_0.05_KM_Banded_20.csv', delimiter=',')
    # banded_KMA = np.loadtxt('debugging_figs/detect_times_0.05_KMA_Banded_20.csv', delimiter=',')

    # print("20 Block vs Cai", wilcoxon(block_ours, block_cai, alternative='less', zero_method='wilcox', correction=False))
    # print("20 Block vs KM", wilcoxon(block_ours, block_KM, alternative='less', zero_method='wilcox', correction=False))
    # print("20 Block vs KMA", wilcoxon(block_ours, block_KMA, alternative='less', zero_method='wilcox', correction=False))
    # print()
    # print("20 Banded vs Cai", wilcoxon(banded_ours, banded_cai, alternative='less', zero_method='wilcox', correction=False))
    # print("20 Banded vs KM", wilcoxon(banded_ours, banded_KM, alternative='less', zero_method='wilcox', correction=False))
    # print("20 Banded vs KMA", wilcoxon(banded_ours, banded_KMA, alternative='less', zero_method='wilcox', correction=False))
    # print()

    # block_ours = np.loadtxt('debugging_figs/detect_times_0.05_Ours_Block_40.csv', delimiter=',')
    # block_cai = np.loadtxt('debugging_figs/detect_times_0.05_Cai_Block_40.csv', delimiter=',')
    # block_KM = np.loadtxt('debugging_figs/detect_times_0.05_KM_Block_40.csv', delimiter=',')
    # block_KMA = np.loadtxt('debugging_figs/detect_times_0.05_KMA_Block_40.csv', delimiter=',')

    # banded_ours = np.loadtxt('debugging_figs/detect_times_0.05_Ours_Banded_40.csv', delimiter=',')
    # banded_cai = np.loadtxt('debugging_figs/detect_times_0.05_Cai_Banded_40.csv', delimiter=',')
    # banded_KM = np.loadtxt('debugging_figs/detect_times_0.05_KM_Banded_40.csv', delimiter=',')
    # banded_KMA = np.loadtxt('debugging_figs/detect_times_0.05_KMA_Banded_40.csv', delimiter=',')

    # print("40 Block vs Cai", wilcoxon(block_ours, block_cai, alternative='less', zero_method='wilcox', correction=False))
    # print("40 Block vs KM", wilcoxon(block_ours, block_KM, alternative='less', zero_method='wilcox', correction=False))
    # print("40 Block vs KMA", wilcoxon(block_ours, block_KMA, alternative='less', zero_method='wilcox', correction=False))
    # print()
    # print("40 Banded vs Cai", wilcoxon(banded_ours, banded_cai, alternative='less', zero_method='wilcox', correction=False))
    # print("40 Banded vs KM", wilcoxon(banded_ours, banded_KM, alternative='less', zero_method='wilcox', correction=False))
    # print("40 Banded vs KMA", wilcoxon(banded_ours, banded_KMA, alternative='less', zero_method='wilcox', correction=False))
    # print()

    # block_ours = np.loadtxt('debugging_figs/detect_times_0.05_Ours_Block_60.csv', delimiter=',')
    # block_cai = np.loadtxt('debugging_figs/detect_times_0.05_Cai_Block_60.csv', delimiter=',')
    # block_KM = np.loadtxt('debugging_figs/detect_times_0.05_KM_Block_60.csv', delimiter=',')
    # block_KMA = np.loadtxt('debugging_figs/detect_times_0.05_KMA_Block_60.csv', delimiter=',')

    # banded_ours = np.loadtxt('debugging_figs/detect_times_0.05_Ours_Banded_60.csv', delimiter=',')
    # banded_cai = np.loadtxt('debugging_figs/detect_times_0.05_Cai_Banded_60.csv', delimiter=',')
    # banded_KM = np.loadtxt('debugging_figs/detect_times_0.05_KM_Banded_60.csv', delimiter=',')
    # banded_KMA = np.loadtxt('debugging_figs/detect_times_0.05_KMA_Banded_60.csv', delimiter=',')

    # print("60 Block vs Cai", wilcoxon(block_ours, block_cai, alternative='less', zero_method='wilcox', correction=False))
    # print("60 Block vs KM", wilcoxon(block_ours, block_KM, alternative='less', zero_method='wilcox', correction=False))
    # print("60 Block vs KMA", wilcoxon(block_ours, block_KMA, alternative='less', zero_method='wilcox', correction=False))
    # print()
    # print("60 Banded vs Cai", wilcoxon(banded_ours, banded_cai, alternative='less', zero_method='wilcox', correction=False))
    # print("60 Banded vs KM", wilcoxon(banded_ours, banded_KM, alternative='less', zero_method='wilcox', correction=False))
    # print("60 Banded vs KMA", wilcoxon(banded_ours, banded_KMA, alternative='less', zero_method='wilcox', correction=False))
    # print()

    # block_ours = np.loadtxt('debugging_figs/detect_times_0.05_Ours_Block_80.csv', delimiter=',')
    # block_cai = np.loadtxt('debugging_figs/detect_times_0.05_Cai_Block_80.csv', delimiter=',')
    # block_KM = np.loadtxt('debugging_figs/detect_times_0.05_KM_Block_80.csv', delimiter=',')
    # block_KMA = np.loadtxt('debugging_figs/detect_times_0.05_KMA_Block_80.csv', delimiter=',')

    # banded_ours = np.loadtxt('debugging_figs/detect_times_0.05_Ours_Banded_80.csv', delimiter=',')
    # banded_cai = np.loadtxt('debugging_figs/detect_times_0.05_Cai_Banded_80.csv', delimiter=',')
    # banded_KM = np.loadtxt('debugging_figs/detect_times_0.05_KM_Banded_80.csv', delimiter=',')
    # banded_KMA = np.loadtxt('debugging_figs/detect_times_0.05_KMA_Banded_80.csv', delimiter=',')

    # print("80 Block vs Cai", wilcoxon(block_ours, block_cai, alternative='less', zero_method='wilcox', correction=False))
    # print("80 Block vs KM", wilcoxon(block_ours, block_KM, alternative='less', zero_method='wilcox', correction=False))
    # print("80 Block vs KMA", wilcoxon(block_ours, block_KMA, alternative='less', zero_method='wilcox', correction=False))
    # print()
    # print("80 Banded vs Cai", wilcoxon(banded_ours, banded_cai, alternative='less', zero_method='wilcox', correction=False))
    # print("80 Banded vs KM", wilcoxon(banded_ours, banded_KM, alternative='less', zero_method='wilcox', correction=False))
    # print("80 Banded vs KMA", wilcoxon(banded_ours, banded_KMA, alternative='less', zero_method='wilcox', correction=False))
    # print()

    # mesonet_ours = np.loadtxt('debugging_figs/detect_times_0.05_Ours_Mesonet.csv', delimiter=',')
    # mesonet_cai = np.loadtxt('debugging_figs/detect_times_0.05_Cai_Mesonet.csv', delimiter=',')
    # mesonet_KM = np.loadtxt('debugging_figs/detect_times_0.05_KM_Mesonet.csv', delimiter=',')
    # mesonet_KMA = np.loadtxt('debugging_figs/detect_times_0.05_KMA_Mesonet.csv', delimiter=',')

    # mesonet_ours_opo = np.loadtxt('debugging_figs/detect_times_0.01_Ours_Mesonet.csv', delimiter=',')
    # mesonet_cai_opo = np.loadtxt('debugging_figs/detect_times_0.01_Cai_Mesonet.csv', delimiter=',')
    # mesonet_KM_opo = np.loadtxt('debugging_figs/detect_times_0.01_KM_Mesonet.csv', delimiter=',')
    # mesonet_KMA_opo = np.loadtxt('debugging_figs/detect_times_0.01_KMA_Mesonet.csv', delimiter=',')

    
    # print("Mesonet vs Cai", wilcoxon(mesonet_ours, mesonet_cai, alternative='less', zero_method='wilcox', correction=False))
    # print("Mesonet vs KM", wilcoxon(mesonet_ours, mesonet_KM, alternative='less', zero_method='wilcox', correction=False))
    # print("Mesonet vs KMA", wilcoxon(mesonet_ours, mesonet_KMA, alternative='less', zero_method='wilcox', correction=False))
    # print()
    # print("Mesonet vs Cai 0.01", wilcoxon(mesonet_ours_opo, mesonet_cai_opo, alternative='less', zero_method='wilcox', correction=False))
    # print("Mesonet vs KM 0.01", wilcoxon(mesonet_ours_opo, mesonet_KM_opo, alternative='less', zero_method='wilcox', correction=False))
    # print("Mesonet vs KMA 0.01", wilcoxon(mesonet_ours_opo, mesonet_KMA_opo, alternative='less', zero_method='wilcox', correction=False))
