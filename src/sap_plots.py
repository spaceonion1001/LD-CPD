import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import glob
import argparse
from scipy.signal import find_peaks

sns.set()
import matplotlib
matplotlib.use('Agg')


def symmetrize_from_vector(a, dim):
    """
    Turns a vector of lower triangular matrix entries into symmetric matrix.
    """
    A = np.zeros((dim, dim))
    ti_1, ti_2 = np.tril_indices(A.shape[0], k=0)
    for idx in range(a.shape[0]):
        val = a[idx]
        i = ti_1[idx]
        j = ti_2[idx]
        A[i, j] = val
    A = A + A.T - np.diag(np.diag(A))
    return A


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clime', action='store_true',
                        help='Use CLIME estimator results for KM/KMA')
    parser.add_argument('--results_path', type=str, default='./results',
                        help='Root results directory')
    parser.add_argument('--data_path', type=str, default='../data',
                        help='Path to data directory containing sap_scaled_returns.csv')
    parser.add_argument('--dim', type=int, default=80,
                        help='Dimension used in SAP run (determines subdirectory name)')
    parser.add_argument('--seed', type=int, default=50,
                        help='Seed used in SAP run (determines subdirectory name)')
    parser.add_argument('--burn_in', type=int, default=200,
                        help='Burn-in steps to skip from LD-CPD results')
    parser.add_argument('--post_window_size', type=int, default=22,
                        help='Post-window offset used to look up end-of-window dates')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    kesh_suffix = '_clime' if args.clime else ''

    # ------------------------------------------------------------------ #
    # Output directory
    # ------------------------------------------------------------------ #
    out_dir = './sap_figs'
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Result subdirectory paths (relative to project root)
    # ------------------------------------------------------------------ #
    sap_subdir = 'sap_500_{}/{}'.format(args.dim, args.seed)

    our_path      = os.path.join(args.results_path, 'sap_results',                               sap_subdir)
    cai_path      = os.path.join(args.results_path, 'sap_results_cai',                           sap_subdir)
    kesh_path     = os.path.join(args.results_path, 'sap_results_kesh{}'.format(kesh_suffix),    sap_subdir)
    kesh_alt_path = os.path.join(args.results_path, 'sap_results_kesh_alt{}'.format(kesh_suffix), sap_subdir)

    # ------------------------------------------------------------------ #
    # Load stock data
    # ------------------------------------------------------------------ #
    stocks = pd.read_csv(os.path.join(args.data_path, 'sap_scaled_returns.csv'), sep='\t')
    stocks = stocks.set_index('Date')

    # ------------------------------------------------------------------ #
    # Load results
    # ------------------------------------------------------------------ #
    cai_results  = pd.read_csv(os.path.join(cai_path,      'global_test_vals.csv'), sep=',', header=None)
    cai_idxs     = pd.read_csv(os.path.join(cai_path,      'chosen_idxs.csv'),      sep=',', header=None)

    kesh_results     = pd.read_csv(os.path.join(kesh_path,     'global_test_vals.csv'), sep=',', header=None)
    kesh_idxs        = pd.read_csv(os.path.join(kesh_path,     'chosen_idxs.csv'),      sep=',', header=None)

    kesh_alt_results = pd.read_csv(os.path.join(kesh_alt_path, 'global_test_vals.csv'), sep=',', header=None)
    kesh_alt_idxs    = pd.read_csv(os.path.join(kesh_alt_path, 'chosen_idxs.csv'),      sep=',', header=None)

    our_results = pd.read_csv(os.path.join(our_path, 'lrt_vals.csv'), sep=',', header=None)
    our_idxs    = pd.read_csv(os.path.join(our_path, 'chosen_idxs.csv'), sep=',', header=None)

    cutoff = our_results.shape[1] // 2
    our_lrt_values = our_results.iloc[args.burn_in:, 0:cutoff]
    our_p_values   = our_results.iloc[args.burn_in:, cutoff:]

    # ------------------------------------------------------------------ #
    # Load LD-CPD precision matrices
    # ------------------------------------------------------------------ #
    our_matrices = {}
    mat_list = glob.glob(os.path.join(our_path, 'matrix*'))
    mat_list = [os.path.basename(x) for x in mat_list]
    mat_list = [x.split('.')[0] for x in mat_list]
    mat_list = sorted([int(x.split('_')[1]) for x in mat_list])
    for val in mat_list:
        curr_mat = np.loadtxt(os.path.join(our_path, 'matrix_{}.csv'.format(val)), delimiter=',')
        our_matrices[val] = curr_mat.copy()

    # ------------------------------------------------------------------ #
    # Date alignment
    # ------------------------------------------------------------------ #
    stocks_adjusted = stocks.iloc[args.burn_in:-args.post_window_size, :]
    dates_list = pd.to_datetime(stocks_adjusted.index)

    # ------------------------------------------------------------------ #
    # Peak detection
    # ------------------------------------------------------------------ #
    kesh_alt_peaks = find_peaks(kesh_alt_results.values.squeeze(), height=1500,  width=10)[0]
    kesh_peaks     = find_peaks(kesh_results.values.squeeze(),     height=4000,  width=10)[0]
    cai_peaks      = find_peaks(cai_results.values.squeeze(),      height=40000, width=2)[0]
    our_peaks      = find_peaks(our_p_values.min(axis=1).values.squeeze() * (-1), height=600, width=10)[0]

    # ------------------------------------------------------------------ #
    # Cluster identification via precision matrices
    # ------------------------------------------------------------------ #
    cluster_ids = [0, 1, 2, 6, 19]
    idxs_mat = {c: np.nonzero(np.any(symmetrize_from_vector(our_matrices[c], dim=args.dim) != 0, axis=0))[0]
                for c in cluster_ids}
    stock_idxs = {c: our_idxs.values.squeeze()[idxs_mat[c]].astype(int) for c in cluster_ids}

    our_p_values_dated = our_p_values.copy()
    our_p_values_dated.index = dates_list
    our_p_values_dated.columns = our_p_values_dated.columns - cutoff
    inv_p_values_dated = -1 * our_p_values_dated

    # ------------------------------------------------------------------ #
    # Helper to save figures
    # ------------------------------------------------------------------ #
    def save_fig(name):
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, name))
        plt.close()

    # ------------------------------------------------------------------ #
    # Plot 1: LD-CPD p-values with LD-CPD changepoints
    # ------------------------------------------------------------------ #
    color_vals = ['crimson', 'limegreen', 'tomato', 'magenta']
    plt.figure(figsize=(35, 10))
    plt.plot(dates_list, our_p_values.values.min(axis=1) * -1, label='LD-CPD', alpha=0.85, linewidth=5)
    plt.ylabel('Inverted Log(P-values)', fontsize=40)
    plt.xticks(rotation=30, fontsize=28)
    plt.yticks(fontsize=28)
    plt.ylim(0.0, 800.0)
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    for i, dval in enumerate(stocks_adjusted.iloc[our_peaks, :].index):
        plt.axvline(pd.to_datetime(dval), color=color_vals[i], linestyle='--', alpha=0.95, label=dval, linewidth=5)
    plt.legend(loc='center left', bbox_to_anchor=(0.83, 0.5), fontsize=35, fancybox=True, shadow=True, ncol=1)
    plt.title("S&P 500 - LD-CPD Log(P-Values) with $\\it{LD\\text{-}CPD}$ Changepoints", fontsize=40)
    save_fig('ldcpd_pvals_ldcpd_cps{}.png'.format(kesh_suffix))

    # ------------------------------------------------------------------ #
    # Plot 2: LD-CPD p-values with KMA changepoints
    # ------------------------------------------------------------------ #
    color_vals = ['crimson', 'limegreen', 'tomato', 'magenta', 'brown']
    plt.figure(figsize=(35, 10))
    plt.plot(dates_list, our_p_values.values.min(axis=1) * -1, label='LD-CPD', alpha=0.85, linewidth=5)
    plt.ylabel('Inverted Log(P-values)', fontsize=40)
    plt.xticks(rotation=30, fontsize=28)
    plt.yticks(fontsize=28)
    plt.ylim(0.0, 800.0)
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    for i, dval in enumerate(stocks_adjusted.iloc[kesh_peaks, :].index):
        plt.axvline(pd.to_datetime(dval), color=color_vals[i], linestyle='--', alpha=0.95, label=dval, linewidth=5)
    plt.legend(loc='center left', bbox_to_anchor=(0.83, 0.5), fontsize=35, fancybox=True, shadow=True, ncol=1)
    plt.title("S&P 500 - LD-CPD Log(P-Values) with $\\it{{KMA}}$ Changepoints", fontsize=40)
    save_fig('ldcpd_pvals_kma_cps{}.png'.format(kesh_suffix))

    # ------------------------------------------------------------------ #
    # Plot 3: LD-CPD p-values with KM changepoints
    # ------------------------------------------------------------------ #
    color_vals = ['crimson', 'limegreen', 'tomato', 'magenta', 'brown', 'orange']
    plt.figure(figsize=(35, 10))
    plt.plot(dates_list, our_p_values.values.min(axis=1) * -1, label='LD-CPD', alpha=0.85, linewidth=5)
    plt.ylabel('Inverted Log(P-values)', fontsize=40)
    plt.xticks(rotation=30, fontsize=28)
    plt.yticks(fontsize=28)
    plt.ylim(0.0, 800.0)
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    for i, dval in enumerate(stocks_adjusted.iloc[kesh_alt_peaks, :].index):
        plt.axvline(pd.to_datetime(dval), color=color_vals[i], linestyle='--', alpha=0.95, label=dval, linewidth=5)
    plt.legend(loc='center left', bbox_to_anchor=(0.83, 0.5), fontsize=35, fancybox=True, shadow=True, ncol=1)
    plt.title("S&P 500 - LD-CPD Log(P-Values) with $\\it{{KM}}$ Changepoints", fontsize=40)
    save_fig('ldcpd_pvals_km_cps{}.png'.format(kesh_suffix))

    # ------------------------------------------------------------------ #
    # Plot 4: LD-CPD p-values with XCC changepoints
    # ------------------------------------------------------------------ #
    color_vals = ['crimson', 'limegreen', 'tomato', 'magenta', 'brown', 'orange', 'yellowgreen', 'purple']
    plt.figure(figsize=(35, 10))
    plt.plot(dates_list, our_p_values.values.min(axis=1) * -1, label='LD-CPD', alpha=0.85, linewidth=5)
    plt.ylabel('Inverted Log(P-values)', fontsize=40)
    plt.xticks(rotation=30, fontsize=28)
    plt.yticks(fontsize=28)
    plt.ylim(0.0, 800.0)
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    for i, dval in enumerate(stocks_adjusted.iloc[cai_peaks, :].index):
        plt.axvline(pd.to_datetime(dval), color=color_vals[i], linestyle='--', alpha=0.95, label=dval, linewidth=5)
    plt.legend(loc='center left', bbox_to_anchor=(0.83, 0.5), fontsize=35, fancybox=True, shadow=True, ncol=1)
    plt.title("S&P 500 - LD-CPD Log(P-Values) with $\\it{{XCC}}$ Changepoints", fontsize=40)
    save_fig('ldcpd_pvals_xcc_cps{}.png'.format(kesh_suffix))

    # ------------------------------------------------------------------ #
    # Plot 5: KMA test statistic with KMA changepoints
    # ------------------------------------------------------------------ #
    color_vals = ['crimson', 'black', 'tomato', 'magenta', 'blue']
    plt.figure(figsize=(35, 10))
    plt.plot(dates_list, kesh_results.values.squeeze(), label='KMA', alpha=1.0, color='green', linewidth=5)
    plt.ylabel('Test Statistic', fontsize=40)
    plt.xticks(rotation=30, fontsize=28)
    plt.yticks(fontsize=28)
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    for i, dval in enumerate(stocks_adjusted.iloc[kesh_peaks, :].index):
        plt.axvline(pd.to_datetime(dval), color=color_vals[i], linestyle='--', alpha=0.95, label=dval, linewidth=5)
    plt.legend(loc='center left', bbox_to_anchor=(0.83, 0.5), fontsize=35, fancybox=True, shadow=True, ncol=1)
    plt.title("S&P 500 - KMA Test Statistic with $\\it{{KMA}}$ Changepoints", fontsize=40)
    save_fig('kma_stat_kma_cps{}.png'.format(kesh_suffix))

    # ------------------------------------------------------------------ #
    # Plot 6: KM test statistic with KM changepoints
    # ------------------------------------------------------------------ #
    color_vals = ['crimson', 'limegreen', 'tomato', 'magenta', 'blue', 'orange']
    plt.figure(figsize=(35, 10))
    plt.plot(dates_list, kesh_alt_results.values.squeeze(), label='KM', alpha=1.0, color='black', linewidth=5)
    plt.ylabel('Test Statistic', fontsize=40)
    plt.xticks(rotation=30, fontsize=28)
    plt.yticks(fontsize=28)
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    for i, dval in enumerate(stocks_adjusted.iloc[kesh_alt_peaks, :].index):
        plt.axvline(pd.to_datetime(dval), color=color_vals[i], linestyle='--', alpha=0.95, label=dval, linewidth=5)
    plt.legend(loc='center left', bbox_to_anchor=(0.83, 0.5), fontsize=35, fancybox=True, shadow=True, ncol=1)
    plt.title("S&P 500 - KM Test Statistic with $\\it{{KM}}$ Changepoints", fontsize=40)
    save_fig('km_stat_km_cps{}.png'.format(kesh_suffix))

    # ------------------------------------------------------------------ #
    # Plot 7: XCC test statistic with XCC changepoints
    # ------------------------------------------------------------------ #
    color_vals = ['green', 'tomato', 'magenta', 'blue', 'orange']
    cai_peaks_plot = stocks_adjusted.iloc[
        [cai_peaks[0], cai_results.values.argmax(), cai_peaks[-2], cai_peaks[-1]], :
    ].index
    plt.figure(figsize=(35, 10))
    plt.plot(dates_list, cai_results.values.squeeze(), label='XCC', alpha=1.0, color='red', linewidth=5)
    plt.ylabel('Test Statistic', fontsize=40)
    plt.xticks(rotation=30, fontsize=28)
    plt.yticks(fontsize=28)
    plt.gca().xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    for i, dval in enumerate(cai_peaks_plot):
        plt.axvline(pd.to_datetime(dval), color=color_vals[i], linestyle='--', alpha=0.95, label=dval, linewidth=4)
    plt.legend(loc='center left', bbox_to_anchor=(0.83, 0.5), fontsize=35, fancybox=True, shadow=True, ncol=1)
    plt.title("S&P 500 - XCC Test Statistic with $\\it{{XCC}}$ Changepoints", fontsize=40)
    save_fig('xcc_stat_xcc_cps{}.png'.format(kesh_suffix))

    # ------------------------------------------------------------------ #
    # Plot 8: Selected cluster p-values (subplots)
    # ------------------------------------------------------------------ #
    cluster_colors = ['blue', 'green', 'magenta', 'orange', 'crimson']
    inv_lim = inv_p_values_dated.loc[:, cluster_ids]
    fig, ax = plt.subplots(len(cluster_ids), 1, figsize=(25, 12), sharex=True, sharey=True)
    for i, (cid, col) in enumerate(zip(cluster_ids, cluster_colors)):
        ax[i].plot(inv_lim.loc[:, cid], color=col, label=cid, linewidth=5)
        ax[i].set_ylim(0.0, 800.0)
        ax[i].tick_params(axis='y', labelsize=16)
    fig.text(0.04, 0.5, 'Inverted Log(P-values)', va='center', rotation='vertical', fontsize=36)
    plt.xticks(rotation=30, fontsize=30)
    ax[0].set_title("Selected S&P 500 Clusters", fontsize=34)
    legend = fig.legend(loc='center left', bbox_to_anchor=(0.86, 0.5), fontsize=40, fancybox=True, shadow=True)
    legend.set_title("Cluster Labels")
    plt.setp(legend.get_title(), fontsize=40)
    plt.xlabel("Date", fontsize=36)
    save_fig('selected_clusters{}.png'.format(kesh_suffix))

    # ------------------------------------------------------------------ #
    # Plot 9: Utility stocks (cluster 1) over time with LD-CPD changepoint
    # ------------------------------------------------------------------ #
    utility_stocks = stocks_adjusted.iloc[:, stock_idxs[1]]
    utility_stocks.index = pd.to_datetime(utility_stocks.index)
    utility_stocks.plot(figsize=(10, 6), legend=True, alpha=1.0)
    plt.xlabel('')
    if len(our_peaks) > 0:
        plt.axvline(stocks_adjusted.index[our_peaks[0]], linestyle='--', color='yellowgreen',
                    label=stocks_adjusted.index[our_peaks[0]], alpha=0.7)
    plt.legend(loc='best', prop={'size': 13}, title='Stocks')
    save_fig('utility_stocks_cluster1{}.png'.format(kesh_suffix))

    print("All figures saved to {}".format(out_dir))
