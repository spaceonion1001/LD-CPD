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

if __name__ == '__main__':
    base_dir = '/home/dink/Documents/Research/Correlation-Changepoint-Detection/amoc_figs'
    for sim in ['orthogonal_small_block', 'orthogonal_cross_block', 'orthogonal_multiple_block', 'orthogonal_hard', 'orthogonal_cross_hard']:
        for dim in [20, 40, 60, 80]:
            block_ours = np.loadtxt(os.path.join(base_dir, '{}/avg_amoc/auc_{}_{}_dim_{}.csv'.format(dim, 'ours', sim, dim)), delimiter=',')
            block_cai = np.loadtxt(os.path.join(base_dir, '{}/avg_amoc/auc_{}_{}_dim_{}.csv'.format(dim, 'cai', sim, dim)), delimiter=',')
            block_kesh = np.loadtxt(os.path.join(base_dir, '{}/avg_amoc/auc_{}_{}_dim_{}.csv'.format(dim, 'kesh', sim, dim)), delimiter=',')
            block_kesh_alt = np.loadtxt(os.path.join(base_dir, '{}/avg_amoc/auc_{}_{}_dim_{}.csv'.format(dim, 'kesh_alt', sim, dim)), delimiter=',')
            print("********************")
            print(sim, dim)
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
            plt.ylabel("TR-AMOC-AUC", fontsize=26)
            #plt.xlabel("FPR", fontsize=26)
            plt.tight_layout()
            plt.savefig('lrt_test_figs/auc_conf_{}_dim_{}.png'.format(sim, dim))
            plt.close()
            # print("Levene All {}".format(levene(block_ours, block_cai, block_kesh, block_kesh_alt, center='mean')))
            # print("Levene XCC {}".format(levene(block_ours, block_cai, center='mean')))
            # print("Levene KMA {}".format(levene(block_ours, block_kesh, center='mean')))
            # print("Levene KM {}".format(levene(block_ours, block_kesh_alt, center='mean')))
            print(">>>>><><><><><><<<<<<<")
            print("VARIANCE TESTS")
            if block_ours.mean() <= block_cai.mean() and block_ours.mean() <= block_kesh.mean() and block_ours.mean() <= block_kesh_alt.mean():
                x = block_ours
                print("Ours {} XCC {} KMA {} KM {}".format(block_ours.mean(), block_cai.mean(), block_kesh.mean(), block_kesh_alt.mean()))
                print("vs Cai", f_test(x, y=block_cai, alt="less"))
                print("vs KMA", f_test(x, y=block_kesh, alt="less"))
                print("vs KM", f_test(x, y=block_kesh_alt, alt="less"))
            elif block_cai.mean() < block_ours.mean() and block_cai.mean() <= block_kesh.mean() and block_cai.mean() <= block_kesh_alt.mean():
                x = block_cai
                print("Ours {} XCC {} KMA {} KM {}".format(block_ours.mean(), block_cai.mean(), block_kesh.mean(), block_kesh_alt.mean()))
                print("Cai vs Ours", f_test(x, y=block_ours, alt="less"))
                print("Cai vs KMA", f_test(x, y=block_kesh, alt="less"))
                print("Cai vs KM", f_test(x, y=block_kesh_alt, alt="less"))
            elif block_kesh.mean() < block_ours.mean() and block_kesh.mean() <= block_cai.mean() and block_kesh.mean() <= block_kesh_alt.mean():
                x = block_kesh
                print("Ours {} XCC {} KMA {} KM {}".format(block_ours.mean(), block_cai.mean(), block_kesh.mean(), block_kesh_alt.mean()))
                print("KMA vs Ours", f_test(x, y=block_ours, alt="less"))
                print("KMA vs Cai", f_test(x, y=block_cai, alt="less"))
                print("KMA vs KM", f_test(x, y=block_kesh_alt, alt="less"))
            else:
                x = block_kesh_alt
                print("Ours {} XCC {} KMA {} KM {}".format(block_ours.mean(), block_cai.mean(), block_kesh.mean(), block_kesh_alt.mean()))
                print("KM vs Ours", f_test(x, y=block_ours, alt="less"))
                print("KM vs Cai", f_test(x, y=block_cai, alt="less"))
                print("KM vs KMA", f_test(x, y=block_kesh, alt="less"))
            print(">>>>><><><><><><<<<<<<")

            df = pd.DataFrame()
            df['LD-CPD'] = block_ours
            df['XCC'] = block_cai
            df['KMA'] = block_kesh
            df['KM'] = block_kesh_alt
            plt.figure(figsize=(12, 9))
            my_pal = {"LD-CPD": "b", "XCC": "red", "KMA":"green", "KM":"grey"}
            b = sns.boxplot(data=df, palette=my_pal)
            if sim == 'orthogonal_small_block':
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
            plt.savefig('lrt_test_figs/auc_boxplot_{}_dim_{}.png'.format(sim, dim))
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
        block_kesh = np.loadtxt(os.path.join(base_dir+'/mesonet', '{}/auc_{}_{}.csv'.format(storm, 'kesh', storm)), delimiter=',')
        block_kesh_alt = np.loadtxt(os.path.join(base_dir+'/mesonet/', '{}/auc_{}_{}.csv'.format(storm, 'kesh_alt', storm)), delimiter=',')
        print("********************")
        print(storm)
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
        plt.savefig('lrt_test_figs/auc_conf_{}.png'.format(storm))
        plt.close()
        # print("Levene {}".format(levene(block_ours, block_cai, block_kesh, block_kesh_alt, center='mean')))
        # print("Levene {}".format(levene(block_ours, block_cai, center='mean')))
        # print("Levene {}".format(levene(block_ours, block_kesh, center='mean')))
        # print("Levene {}".format(levene(block_ours, block_kesh_alt, center='mean')))
        x = block_ours
        print("Ours {} XCC {} KMA {} KM {}".format(block_ours.mean(), block_cai.mean(), block_kesh.mean(), block_kesh_alt.mean()))
        print("vs Cai", f_test(x, y=block_cai, alt="less"))
        print("vs KMA", f_test(x, y=block_kesh, alt="less"))
        print("vs KM", f_test(x, y=block_kesh_alt, alt="less"))

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
        plt.savefig('lrt_test_figs/auc_boxplot_{}.png'.format(storm))
        plt.close()
        print("********************")
        print()
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
