import numpy as np
import nibabel as nib
import pickle
from utils import get_DTs, lag_pearsonr, save_nii, tj_fit, FDR_p, ev_annot_freq, hrf_convolution, nearest_peak
from s_light import get_vox_map
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

data_fpath = '../data/'
header_fpath = data_fpath + 'header.nii'
non_nan = nib.load(data_fpath + 'valid_vox.nii').get_fdata().T > 0
SL_allvox = pickle.load(open(data_fpath + 'SL/SL_allvox.p', 'rb'))

lag_corr = pickle.load(open('../outputs/AUC_S10_jointfit_lag_corr.p', 'rb'))
nSL = len(lag_corr)
max_lag = 10
nTR = 60
TR = 1.5

peaklag = nSL*[None]
for sl_i in range(nSL):
    peaklag[sl_i] = np.full((6,100), np.nan)
    for rep in range(6):
        for p in range(100):
            peaklag[sl_i][rep,p] = nearest_peak(lag_corr[sl_i][rep,:,p])


peaklagdiff, peaklagdiff_q = get_vox_map([TR*(sl[1:,:].mean(0)-sl[0,:]) for sl in peaklag], SL_allvox, non_nan)
save_nii('../outputs/peaklagdiff.nii', header_fpath, peaklagdiff)
save_nii('../outputs/peaklagdiff_q.nii', header_fpath, peaklagdiff_q)

# peaklag = nSL*[None]
# for sl_i in range(nSL):
#     peaklag[sl_i] = np.full((6,100), np.nan)
#     for rep in range(6):
#         for p in range(100):
#             lag = max_lag
#             while (lag >= 1 and lag <= 2*max_lag - 1):
#                 win = lag_corr[sl_i][rep,(lag-1):(lag+2),p]
#                 if (win[1] > win[0]) and (win[1] > win[2]):
#                     break
#                 if win[0] > win[2]:
#                     lag -= 1
#                 else:
#                     lag += 1
#             peaklag[sl_i][rep,p] = lag


# peaklagdiff, peaklagdiff_q = get_vox_map([sl[1:,:].mean(0)-sl[0,:] for sl in peaklag], SL_allvox, non_nan)
# save_nii('../outputs/peaklagdiff.nii', header_fpath, peaklagdiff)
# save_nii('../outputs/peaklagdiff_q.nii', header_fpath, peaklagdiff_q)

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
#ev_conv = hrf_convolution(ev_annot_freq())
for sl_i in range(len(SLs)):
    # pick_data = pickle.load(open('../outputs/perm/pickles/_5_' +str(SLs[sl_i])+'_0_100_.p', 'rb'))
    # sl_DT = np.zeros((6, nTR-1))
    # plt.figure()
    # for rep in range(6):
    #     sl_DT[rep,:] = get_DTs(pick_data[1][0][rep])
    #     plt.plot(sl_DT[rep,:])
    # plt.plot(ev_conv)
    # plt.legend(np.arange(6))
    # plt.savefig(str(SLs[sl_i])+'_DT.png', dpi=200)
    # plt.show()

    plt.figure(figsize=(5,5))
    xr = [10-xlim[sl_i,1]-1, 10-xlim[sl_i,0]]
    for rep in range(6):
        plt.plot(np.arange(xlim[sl_i,1]*1.5,(xlim[sl_i,0]-1)*1.5,step=-1.5),lag_corr[SLs[sl_i]][rep,xr[0]:xr[1],0],colors[rep])
    plt.legend(np.arange(6))
    plt.xlim(xlim[sl_i,0]*1.5,xlim[sl_i,1]*1.5)
    plt.plot([xlim[sl_i,0]*1.5,xlim[sl_i,1]*1.5], [0, 0], color='k')
    plt.plot([0, 0], [ylim[sl_i,0], ylim[sl_i,1]], color='k')
    plt.xlabel('Lag (sec)')
    plt.ylabel('Correlation (r)')
    plt.ylim(ylim[sl_i,0], ylim[sl_i,1])
    plt.yticks(np.arange(ylim[sl_i,0], ylim[sl_i,1],0.1))
    plt.tight_layout()
    plt.title(str(sl_i))
    plt.savefig('../outputs/' + str(SLs[sl_i])+'.png', dpi=200)
    plt.show()



# sigcoords = np.transpose(np.where(peaklagdiff_q0vAll[non_nan] < 0.05))
# overlap = np.array([len(np.intersect1d(sigcoords, sl, assume_unique=True)) for sl in SL_allvox])
# overlap_SL = np.where(overlap>7)[0]
# SLind = np.zeros(non_nan.sum())
# for sl in overlap_SL:
#     SLind[SL_allvox[sl]] = sl

# SLind3d = np.full(non_nan.shape, np.nan)
# SLind3d[non_nan] = SLind
# save_nii('allSL.nii', header_fpath, SLind3d)

# lag_corr[sl_i][rep,:,p] = lag_pearsonr(sl_DT[sl_i][rep,:,p], ev_conv[1:], max_lag)

# peaklagdiff0v5, peaklagdiff_q0v5, peaklagdiff_z0v5 = get_vox_map([sl[5,:]-sl[0,:] for sl in peaklag], SL_allvox, non_nan, return_q=True, return_z=True)
# save_nii('peaklagdiff0v5_q.nii', header_fpath, peaklagdiff_q0v5)
# peaklagdiff0vAll, peaklagdiff_q0vAll, peaklagdiff_z0vAll = get_vox_map([sl[1:,:].mean(0)-sl[0,:] for sl in peaklag], SL_allvox, non_nan, return_q=True, return_z=True)
# save_nii('peaklagdiff0vAll_q.nii', header_fpath, peaklagdiff_q0vAll)

# lin_q = peaklagdiff_q[non_nan]
# mincoords = np.transpose(np.where(lin_q==lin_q.min()))
# overlap = [len(np.intersect1d(mincoords, sl, assume_unique=True)) for sl in SL_allvox]

# # SL 1479
# SLind = np.zeros(non_nan.sum())
# SLind[SL_allvox[1479]] = 1
# SLind3d = np.full(non_nan.shape, np.nan)
# SLind3d[non_nan] = SLind
# save_nii('SL1479.nii', header_fpath, SLind3d)



# clust = nib.load('../outputs/AUC_S10_jointfit_mean_ward.nii').get_fdata().T

# lag0_diff = nSL*[None]
# maxlag_diff = nSL*[None]
# for sl_i in range(nSL):
#     lag0_diff[sl_i] = lag_corr[sl_i][0,max_lag,:] - lag_corr[sl_i][1:,max_lag,:].mean(0)
#     maxlag = np.argmax(lag_corr[sl_i], axis=1)
#     maxlag_diff[sl_i] = maxlag[0,:] - maxlag[1:,:].mean(0)


# peaklag3d = get_vox_map(peaklag, SL_allvox, non_nan, return_q=False, return_z=False)
# save_nii('peaklag3d_0.nii', header_fpath, peaklag3d[:,:,:,0])
# save_nii('peaklag3d_5.nii', header_fpath, peaklag3d[:,:,:,5])

# z = np.zeros(nSL)
# for sl_i in range(nSL):
#     pldiff = peaklag[sl_i][0,:] - peaklag[sl_i][5,:]
#     z[sl_i] = (pldiff[0] - pldiff[1:].mean())/np.std(pldiff[1:])

# peaklagdiff, peaklagdiff_q, peaklagdiff_z = get_vox_map([sl[5,:]-sl[0,:] for sl in peaklag], SL_allvox, non_nan, return_q=True, return_z=True)
# nClust = int(np.nanmax(clust)+1)
# for c in range(nClust):
#     clust_perm = maxlagdiff_3d[clust==c].mean(0)
#     print(c,(clust_perm[0]-clust_perm[1:].mean())/np.std(clust_perm[1:]), maxlagdiff_zvals[clust==c].mean())

# #save_nii('peaklagdiff_q.nii', header_fpath, peaklagdiff_q)

# maxlagdiff_3d = get_vox_map([sl[:,np.newaxis] for sl in maxlag_diff], SL_allvox, non_nan, return_q = False, return_z=False)
# maxlagdiff_3d, maxlagdiff_zvals = get_vox_map(maxlag_diff, SL_allvox, non_nan_mask, return_q=False, return_z=True)

# pld = get_vox_map([(sl[5,:]-sl[0,:])[:,np.newaxis] for sl in peaklag], SL_allvox, non_nan, return_q=False, return_z=False)
# nClust = int(np.nanmax(clust)+1)
# for c in range(nClust):
#     clust_perm = pld[clust==c].mean(0)
#     print(c,(clust_perm[0]-clust_perm[1:].mean())/np.std(clust_perm[1:]))



# corrcorr = nSL * [None]
# for sl_i in range(nSL):
#     corrcorr[sl_i] = np.full(100, np.nan)
#     for p in range(100):
#         corrcorr[sl_i][p] = pearsonr(lag_corr[sl_i][0,:,p],lag_corr[sl_i][5,:,p])[0]
# a = np.row_stack(corrcorr)
# z = (a[:,1:].mean(1)-a[:,0])/np.std(a[:,1:],axis=1)

# cc = get_vox_map(corrcorr, SL_allvox, non_nan, return_q=True, return_z=True)

# get_vox_map([sl[:,np.newaxis] for sl in maxlag_diff], SL_allvox, non_nan, return_q = False, return_z=False)



# lag0_diff = nSL*[None]
# for sl_i in range(nSL):
#     lag0_diff[sl_i] = lag_corr[sl_i][0,max_lag,:] - lag_corr[sl_i][5,max_lag,:]
# ld = get_vox_map(lag0_diff, SL_allvox, non_nan, return_q=True, return_z=True)
# for c in range(nClust):
#     clust_perm = ld[0][clust==c]
#     print(c,(clust_perm[0]-clust_perm[1:].mean())/np.std(clust_perm[1:]))

# def lag_corr(dataset, roi_clusters, ev_conv, max_lag,
#              header_fpath, save_prefix):
#     """Lag correlation between HMM events and hand-annotated event data

#     Refits an HMM within each cluster of roi_clusters, and computes a lag
#     correlation between the derivative of the expected value of the event
#     and ev_conv.

#     Parameters
#     ----------
#     dataset : Dataset
#         Data and mask for searchlight analysis
#     roi_clusters : ndarray
#         Mask of ROI clusters.
#     ev_conv : ndarray
#         Event boundaries convolved with the HRF
#     max_lag : int
#         Maximum lag to compute correlations for
#     header_fpath : string
#         File to use as a nifti template
#     save_prefix : string
#         Partial path for saving lag-0 maps

#     Returns
#     -------
#     first_lagcorr : ndarray
#         Lag correlation for first viewing (121 x 145 x 121 x (1+ 2*max_lag))
#     lasts_lagcorr : ndarray
#         Lag correlation for later viewings (121 x 145 x 121 x (1+ 2*max_lag))
#     """

#     vox_map_shape = (121, 145, 121, 1 + 2*max_lag)
#     n_rois = int(np.max(roi_clusters))

#     first_lagcorr = np.full(shape=vox_map_shape, fill_value=np.nan)
#     lasts_lagcorr = np.full(shape=vox_map_shape, fill_value=np.nan)

#     print('Computing lag correlations for ' + str(n_rois) + ' clusters')
#     for cluster in range(1, n_rois + 1):
#         mask = roi_clusters == cluster

#         cluster_data = dataset.data[:, :, mask]
#         segs = tj_fit(cluster_data)

#         dts_first = get_DTs(segs[0])
#         dts_lasts = get_DTs(segs[1])

#         first_lagcorr[mask] = lag_pearsonr(dts_first, ev_conv[1:], max_lag)
#         lasts_lagcorr[mask] = lag_pearsonr(dts_lasts, ev_conv[1:], max_lag)

#     # Save lag-0 results
#     save_nii(save_prefix + '_first.nii', header_fpath,
#              first_lagcorr[:,:,:,max_lag])
#     save_nii(save_prefix + '_lasts.nii', header_fpath,
#              lasts_lagcorr[:,:,:,max_lag])
#     save_nii(save_prefix + '_diff.nii', header_fpath,
#              first_lagcorr[:,:,:,max_lag] - lasts_lagcorr[:,:,:,max_lag])

#     return first_lagcorr, lasts_lagcorr
