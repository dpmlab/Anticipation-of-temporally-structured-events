import numpy as np
import nibabel as nib
import pickle
from utils import get_DTs, lag_pearsonr, save_nii, tj_fit, FDR_p, ev_annot_freq, hrf_convolution, nearest_peak
from s_light import get_vox_map
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from numpy.random import default_rng
import matplotlib.ticker as plticker
import matplotlib

data_fpath = '../data/'
header_fpath = data_fpath + 'header.nii'

max_lag = 10
nTR = 60
TR = 1.5
nSL = 5354


non_nan = nib.load(data_fpath + 'valid_vox.nii').get_fdata().T > 0
SL_allvox = pickle.load(open(data_fpath + 'SL/SL_allvox.p', 'rb'))
lag_corr = pickle.load(open('../outputs/AUC_S10_jointfit_lag_corr.p', 'rb'))
peaklag = nSL*[None]
for sl_i in range(nSL):
    peaklag[sl_i] = np.full((6,100), np.nan)
    for rep in range(6):
        for p in range(100):
            peaklag[sl_i][rep,p] = nearest_peak(lag_corr[sl_i][rep,:,p])


peaklagdiff, peaklagdiff_q = get_vox_map([TR*(sl[1:,:].mean(0)-sl[0,:]) for sl in peaklag], SL_allvox, non_nan)
save_nii('../outputs/peaklagdiff.nii', header_fpath, peaklagdiff)
save_nii('../outputs/peaklagdiff_q.nii', header_fpath, peaklagdiff_q)

# LO = 2614
# STS = 1479
# PHC = 1054
SLs = [2614, 1479, 1054]
xlim = np.array([[-6, 2,],
                 [-7, 7],
                 [-5, 3]])
ylim = np.array([[-0.1, 0.2],
                 [-0.2, 0.6],
                 [-0.2, 0.6]])
colors = ['#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#b10026']
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
for sl_i in range(len(SLs)):
    pick_data = pickle.load(open('../outputs/perm/pickles/_5_' + str(SLs[sl_i]) + '_0_100_.p', 'rb'))
    sl_DT = []
    for rep in range(6):
        sl_DT.append(get_DTs(pick_data[1][0][rep]))

    nBoot = 100
    bootstrap_rng = default_rng(0)
    boot_lag = np.zeros((nBoot, 6, 2*max_lag+1))
    boot_peak = np.zeros((nBoot, 6))
    for b in range(nBoot):
        ev_conv = hrf_convolution(ev_annot_freq(bootstrap_rng))
        for rep in range(6):
            boot_lag[b,rep,:] = lag_pearsonr(sl_DT[rep], ev_conv[1:], max_lag)
            boot_peak[b,rep] = nearest_peak(boot_lag[b,rep,:])
    CI_init = TR*(max_lag - np.sort(boot_peak[:,0])[[5,95-1]])
    CI_rep = TR*(max_lag - np.sort(boot_peak[:,1:].mean(1))[[5,95-1]])

    print(str(SLs[sl_i]) + ': First Peak CI =' + str(CI_init) + ', Rep Peak CI = ' + str(CI_rep))

    print('Pred (sec): ' + str(TR/(7-1) * (pick_data[0][0][1:].mean() - pick_data[0][0][0])))

    ev_conv = hrf_convolution(ev_annot_freq())
    lag_corr = np.zeros((6, 1 + 2*max_lag))
    for rep in range(6):
        lag_corr[rep,:] = lag_pearsonr(sl_DT[rep], ev_conv[1:], max_lag)

    plt.figure(figsize=(5,5))
    xr = [10-xlim[sl_i,1], 10-xlim[sl_i,0]+1]
    for rep in range(6):
        plt.plot(np.arange(xlim[sl_i,1]*1.5,(xlim[sl_i,0]-1)*1.5,step=-1.5),lag_corr[rep,xr[0]:xr[1]],colors[rep])
    plt.legend(np.arange(6))
    plt.xlim(xlim[sl_i,0]*1.5,xlim[sl_i,1]*1.5)
    plt.plot([xlim[sl_i,0]*1.5,xlim[sl_i,1]*1.5], [0, 0], color='k')
    plt.plot([0, 0], [ylim[sl_i,0], ylim[sl_i,1]], color='k')
    plt.xlabel('Lag (sec)')
    plt.ylabel('Correlation (r)')
    plt.ylim(ylim[sl_i,0], ylim[sl_i,1])
    plt.gca().xaxis.set_major_locator(plticker.MultipleLocator(base=2))
    plt.yticks(np.arange(ylim[sl_i,0], ylim[sl_i,1],0.1))
    plt.tight_layout()
    plt.title(str(sl_i))
    plt.savefig('../outputs/' + str(SLs[sl_i])+'.png', dpi=200)
    plt.savefig('../outputs/' + str(SLs[sl_i])+'.svg')
    #plt.show()
