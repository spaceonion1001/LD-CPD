import numpy as np
from scipy.stats import wilcoxon


if __name__ == '__main__':
    block_ours = np.loadtxt('debugging_figs/detect_times_0.05_Ours_Block_20.csv', delimiter=',')
    block_cai = np.loadtxt('debugging_figs/detect_times_0.05_Cai_Block_20.csv', delimiter=',')
    block_KM = np.loadtxt('debugging_figs/detect_times_0.05_KM_Block_20.csv', delimiter=',')
    block_KMA = np.loadtxt('debugging_figs/detect_times_0.05_KMA_Block_20.csv', delimiter=',')

    
    banded_ours = np.loadtxt('debugging_figs/detect_times_0.05_Ours_Banded_20.csv', delimiter=',')
    banded_cai = np.loadtxt('debugging_figs/detect_times_0.05_Cai_Banded_20.csv', delimiter=',')
    banded_KM = np.loadtxt('debugging_figs/detect_times_0.05_KM_Banded_20.csv', delimiter=',')
    banded_KMA = np.loadtxt('debugging_figs/detect_times_0.05_KMA_Banded_20.csv', delimiter=',')

    print("20 Block vs Cai", wilcoxon(block_ours, block_cai, alternative='less', zero_method='wilcox', correction=False))
    print("20 Block vs KM", wilcoxon(block_ours, block_KM, alternative='less', zero_method='wilcox', correction=False))
    print("20 Block vs KMA", wilcoxon(block_ours, block_KMA, alternative='less', zero_method='wilcox', correction=False))
    print()
    print("20 Banded vs Cai", wilcoxon(banded_ours, banded_cai, alternative='less', zero_method='wilcox', correction=False))
    print("20 Banded vs KM", wilcoxon(banded_ours, banded_KM, alternative='less', zero_method='wilcox', correction=False))
    print("20 Banded vs KMA", wilcoxon(banded_ours, banded_KMA, alternative='less', zero_method='wilcox', correction=False))
    print()

    block_ours = np.loadtxt('debugging_figs/detect_times_0.05_Ours_Block_40.csv', delimiter=',')
    block_cai = np.loadtxt('debugging_figs/detect_times_0.05_Cai_Block_40.csv', delimiter=',')
    block_KM = np.loadtxt('debugging_figs/detect_times_0.05_KM_Block_40.csv', delimiter=',')
    block_KMA = np.loadtxt('debugging_figs/detect_times_0.05_KMA_Block_40.csv', delimiter=',')

    banded_ours = np.loadtxt('debugging_figs/detect_times_0.05_Ours_Banded_40.csv', delimiter=',')
    banded_cai = np.loadtxt('debugging_figs/detect_times_0.05_Cai_Banded_40.csv', delimiter=',')
    banded_KM = np.loadtxt('debugging_figs/detect_times_0.05_KM_Banded_40.csv', delimiter=',')
    banded_KMA = np.loadtxt('debugging_figs/detect_times_0.05_KMA_Banded_40.csv', delimiter=',')

    print("40 Block vs Cai", wilcoxon(block_ours, block_cai, alternative='less', zero_method='wilcox', correction=False))
    print("40 Block vs KM", wilcoxon(block_ours, block_KM, alternative='less', zero_method='wilcox', correction=False))
    print("40 Block vs KMA", wilcoxon(block_ours, block_KMA, alternative='less', zero_method='wilcox', correction=False))
    print()
    print("40 Banded vs Cai", wilcoxon(banded_ours, banded_cai, alternative='less', zero_method='wilcox', correction=False))
    print("40 Banded vs KM", wilcoxon(banded_ours, banded_KM, alternative='less', zero_method='wilcox', correction=False))
    print("40 Banded vs KMA", wilcoxon(banded_ours, banded_KMA, alternative='less', zero_method='wilcox', correction=False))
    print()

    block_ours = np.loadtxt('debugging_figs/detect_times_0.05_Ours_Block_60.csv', delimiter=',')
    block_cai = np.loadtxt('debugging_figs/detect_times_0.05_Cai_Block_60.csv', delimiter=',')
    block_KM = np.loadtxt('debugging_figs/detect_times_0.05_KM_Block_60.csv', delimiter=',')
    block_KMA = np.loadtxt('debugging_figs/detect_times_0.05_KMA_Block_60.csv', delimiter=',')

    banded_ours = np.loadtxt('debugging_figs/detect_times_0.05_Ours_Banded_60.csv', delimiter=',')
    banded_cai = np.loadtxt('debugging_figs/detect_times_0.05_Cai_Banded_60.csv', delimiter=',')
    banded_KM = np.loadtxt('debugging_figs/detect_times_0.05_KM_Banded_60.csv', delimiter=',')
    banded_KMA = np.loadtxt('debugging_figs/detect_times_0.05_KMA_Banded_60.csv', delimiter=',')

    print("60 Block vs Cai", wilcoxon(block_ours, block_cai, alternative='less', zero_method='wilcox', correction=False))
    print("60 Block vs KM", wilcoxon(block_ours, block_KM, alternative='less', zero_method='wilcox', correction=False))
    print("60 Block vs KMA", wilcoxon(block_ours, block_KMA, alternative='less', zero_method='wilcox', correction=False))
    print()
    print("60 Banded vs Cai", wilcoxon(banded_ours, banded_cai, alternative='less', zero_method='wilcox', correction=False))
    print("60 Banded vs KM", wilcoxon(banded_ours, banded_KM, alternative='less', zero_method='wilcox', correction=False))
    print("60 Banded vs KMA", wilcoxon(banded_ours, banded_KMA, alternative='less', zero_method='wilcox', correction=False))
    print()

    block_ours = np.loadtxt('debugging_figs/detect_times_0.05_Ours_Block_80.csv', delimiter=',')
    block_cai = np.loadtxt('debugging_figs/detect_times_0.05_Cai_Block_80.csv', delimiter=',')
    block_KM = np.loadtxt('debugging_figs/detect_times_0.05_KM_Block_80.csv', delimiter=',')
    block_KMA = np.loadtxt('debugging_figs/detect_times_0.05_KMA_Block_80.csv', delimiter=',')

    banded_ours = np.loadtxt('debugging_figs/detect_times_0.05_Ours_Banded_80.csv', delimiter=',')
    banded_cai = np.loadtxt('debugging_figs/detect_times_0.05_Cai_Banded_80.csv', delimiter=',')
    banded_KM = np.loadtxt('debugging_figs/detect_times_0.05_KM_Banded_80.csv', delimiter=',')
    banded_KMA = np.loadtxt('debugging_figs/detect_times_0.05_KMA_Banded_80.csv', delimiter=',')

    print("80 Block vs Cai", wilcoxon(block_ours, block_cai, alternative='less', zero_method='wilcox', correction=False))
    print("80 Block vs KM", wilcoxon(block_ours, block_KM, alternative='less', zero_method='wilcox', correction=False))
    print("80 Block vs KMA", wilcoxon(block_ours, block_KMA, alternative='less', zero_method='wilcox', correction=False))
    print()
    print("80 Banded vs Cai", wilcoxon(banded_ours, banded_cai, alternative='less', zero_method='wilcox', correction=False))
    print("80 Banded vs KM", wilcoxon(banded_ours, banded_KM, alternative='less', zero_method='wilcox', correction=False))
    print("80 Banded vs KMA", wilcoxon(banded_ours, banded_KMA, alternative='less', zero_method='wilcox', correction=False))
    print()

    mesonet_ours = np.loadtxt('debugging_figs/detect_times_0.05_Ours_Mesonet.csv', delimiter=',')
    mesonet_cai = np.loadtxt('debugging_figs/detect_times_0.05_Cai_Mesonet.csv', delimiter=',')
    mesonet_KM = np.loadtxt('debugging_figs/detect_times_0.05_KM_Mesonet.csv', delimiter=',')
    mesonet_KMA = np.loadtxt('debugging_figs/detect_times_0.05_KMA_Mesonet.csv', delimiter=',')

    mesonet_ours_opo = np.loadtxt('debugging_figs/detect_times_0.01_Ours_Mesonet.csv', delimiter=',')
    mesonet_cai_opo = np.loadtxt('debugging_figs/detect_times_0.01_Cai_Mesonet.csv', delimiter=',')
    mesonet_KM_opo = np.loadtxt('debugging_figs/detect_times_0.01_KM_Mesonet.csv', delimiter=',')
    mesonet_KMA_opo = np.loadtxt('debugging_figs/detect_times_0.01_KMA_Mesonet.csv', delimiter=',')

    
    print("Mesonet vs Cai", wilcoxon(mesonet_ours, mesonet_cai, alternative='less', zero_method='wilcox', correction=False))
    print("Mesonet vs KM", wilcoxon(mesonet_ours, mesonet_KM, alternative='less', zero_method='wilcox', correction=False))
    print("Mesonet vs KMA", wilcoxon(mesonet_ours, mesonet_KMA, alternative='less', zero_method='wilcox', correction=False))
    print()
    print("Mesonet vs Cai 0.01", wilcoxon(mesonet_ours_opo, mesonet_cai_opo, alternative='less', zero_method='wilcox', correction=False))
    print("Mesonet vs KM 0.01", wilcoxon(mesonet_ours_opo, mesonet_KM_opo, alternative='less', zero_method='wilcox', correction=False))
    print("Mesonet vs KMA 0.01", wilcoxon(mesonet_ours_opo, mesonet_KMA_opo, alternative='less', zero_method='wilcox', correction=False))
