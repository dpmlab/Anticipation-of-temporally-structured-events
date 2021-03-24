from copy import deepcopy
import time
import numpy as np
from scipy.spatial.distance import cdist
import tables
import glob
import pickle
from tqdm import tqdm
from utils import get_AUCs, tj_fit, save_nii, hyperalign, heldout_ll, FDR_p, get_DTs, ev_annot_freq, hrf_convolution, lag_pearsonr, nearest_peak
from scipy.stats import norm, spearmanr

def get_s_lights(coords, stride=5, radius=5, min_vox=20):
    SL_allvox = []

    for x in range(0, np.max(coords, axis=0)[0] + stride, stride):
        for y in range(0, np.max(coords, axis=0)[1] + stride, stride):
            for z in range(0, np.max(coords, axis=0)[2] + stride, stride):
                dists = cdist(coords, np.array([[x, y, z]]))[:, 0]
                SL_vox = np.where(dists <= radius)[0]
                if len(SL_vox) >= min_vox:
                    SL_allvox.append(SL_vox)

    return SL_allvox

def one_sl(subj_list, subjects, avg_lasts, tune_K, SRM_features, n_events=7):
    if tune_K:
        K_range = np.arange(2, 10)
        ll = np.zeros(len(K_range))
        split = np.array([('predtrw01' in s) for s in subjects])
        rep1 = np.array([d[0] for d in subj_list])
        for i, K in enumerate(K_range):
            ll[i] = heldout_ll(rep1, K, split)
        n_events = K_range[np.argmax(ll)]

    if SRM_features == 0:
        group_data = np.mean(subj_list, axis=0)
    else:
        hyp_data = hyperalign(subj_list, nFeatures=SRM_features)
        group_data = np.mean(hyp_data, axis=0)

    seg = tj_fit(group_data, n_events=n_events, avg_lasts=avg_lasts)
    
    return (get_AUCs(seg), n_events, seg)

def one_sl_shift(subj_list, max_shift):
    group_data = np.mean(subj_list, axis=0).mean(2) # Rep x TR
    rep1 = group_data[0,:]
    rep2_6 = group_data[1:,:].mean(0)

    shift_corr = lag_pearsonr(rep1, rep2_6, max_shift)

    return shift_corr


def one_sl_SF(Intact_subj_list, SFix_subj_list, avg_lasts, SRM_features):
    AUC_diff_Intact = []
    AUC_diff_SFix = []
    for group in Intact_subj_list:
        if SRM_features == 0:
            group_Intact = np.mean(Intact_subj_list[group], axis=0)
            group_SFix = np.mean(SFix_subj_list[group], axis=0)
        else:
            subj_list = []
            for s in range(len(Intact_subj_list[group])):
                subj_list.append(np.concatenate((Intact_subj_list[group][s],SFix_subj_list[group][s]), axis=0))
            hyp_data = hyperalign(subj_list, nFeatures=SRM_features)
            group_data = np.mean(hyp_data, axis=0)
            group_Intact = group_data[:6]
            group_SFix = group_data[6:]

        AUC_Intact = get_AUCs(tj_fit(group_Intact, avg_lasts=avg_lasts))
        AUC_SFix = get_AUCs(tj_fit(group_SFix, avg_lasts=avg_lasts))

        AUC_diff_Intact.append(AUC_Intact - AUC_Intact[0])
        AUC_diff_SFix.append(AUC_SFix - AUC_SFix[0])

    return (np.mean(AUC_diff_Intact, axis=0),
            np.mean(AUC_diff_SFix, axis=0))

def load_pickle(analysis_type, pickle_path, non_nan_mask, SL_allvox, header_fpath, savename):
    nSL = 5354
    nPerm = 100
    TR = 1.5
    nTR = 60
    nEvents = 7
    max_lag = 10

    sl_AUCdiffs = nSL*[None]
    sl_K = nSL*[None]
    sl_AUCdiffs_Intact = nSL*[None]
    sl_AUCdiffs_SFix = nSL*[None]
    sl_DT = nSL*[None]
    sl_shift_corr = nSL*[None]
    for sl_i in range(nSL):
        if analysis_type == 0 or analysis_type == 1 or analysis_type == 3:
            sl_AUCdiffs[sl_i] = np.zeros(nPerm)
        elif analysis_type == 2 or analysis_type == 5 or analysis_type == 7:
            sl_AUCdiffs[sl_i] = np.zeros((5, nPerm))

        if analysis_type == 3:
            sl_K[sl_i] = np.zeros(nPerm)

        if analysis_type == 4:
            sl_AUCdiffs_Intact[sl_i] = np.zeros(nPerm)
            sl_AUCdiffs_SFix[sl_i] = np.zeros(nPerm)

        if analysis_type == 5:
            sl_DT[sl_i] = np.zeros((6,nTR-1,nPerm))

        if analysis_type == 6:
            sl_AUCdiffs_Intact[sl_i] = np.zeros((5, nPerm))
            sl_AUCdiffs_SFix[sl_i] = np.zeros((5, nPerm))

        if analysis_type == 8:
            sl_shift_corr[sl_i] = np.zeros((2*max_lag+1, nPerm))

    pickles = glob.glob(pickle_path + '*.p')
    for i, pick in enumerate(pickles):
        pick_parts = pick.split('_')
        analysis_i = int(pick_parts[1])
        sl_i = int(pick_parts[2])

        if analysis_i == analysis_type:
            with open (pick, 'rb') as fp:
                pick_data = pickle.load(fp)
            for perm_i in range(100):
                if analysis_type == 0 or analysis_type == 1:
                    sl_AUCdiffs[sl_i][perm_i] = TR/(nEvents-1) * (pick_data[perm_i][1]-pick_data[perm_i][0])
                elif analysis_type == 2 or analysis_type == 7:
                    sl_AUCdiffs[sl_i][:,perm_i] = TR/(nEvents-1) * (pick_data[perm_i][1:]-pick_data[perm_i][0])
                elif analysis_type == 3:
                    K = pick_data[1][perm_i]
                    sl_K[sl_i][perm_i] = K
                    sl_AUCdiffs[sl_i][perm_i] = TR/(K-1) * (pick_data[0][perm_i][1:]-pick_data[0][perm_i][0])
                elif analysis_type == 4:
                    sl_AUCdiffs_Intact[sl_i][perm_i] = TR/(nEvents-1) * pick_data[0][perm_i][1]
                    sl_AUCdiffs_SFix[sl_i][perm_i] = TR/(nEvents-1) * pick_data[1][perm_i][1]
                elif analysis_type == 5:
                    sl_AUCdiffs[sl_i][:,perm_i] = TR/(nEvents-1) * (pick_data[0][perm_i][1:]-pick_data[0][perm_i][0])
                    for rep in range(6):
                        sl_DT[sl_i][rep,:,perm_i] = get_DTs(pick_data[1][perm_i][rep])
                elif analysis_type == 6:
                    sl_AUCdiffs_Intact[sl_i][:,perm_i] = TR/(nEvents-1) * pick_data[0][perm_i][1:]
                    sl_AUCdiffs_SFix[sl_i][:,perm_i] = TR/(nEvents-1) * pick_data[1][perm_i][1:]
                elif analysis_type == 8:
                    sl_shift_corr[sl_i][:,perm_i] = pick_data[perm_i][:]

    # Event seg analysis
    if analysis_type == 5:
        ev_conv = hrf_convolution(ev_annot_freq())
        lag_corr = nSL*[None]
        for sl_i in tqdm(range(nSL)):
            lag_corr[sl_i] = np.zeros((6, 1 + 2*max_lag, nPerm))
            for p in range(nPerm):
                for rep in range(6):
                    lag_corr[sl_i][rep,:,p] = lag_pearsonr(sl_DT[sl_i][rep,:,p], ev_conv[1:], max_lag)

        with open(savename + '_lag_corr.p', 'wb') as fp:
            pickle.dump(lag_corr, fp)


    # if analysis_type == 0 or analysis_type == 1 or analysis_type == 3:
    #     vox3d, qvals = get_vox_map(sl_AUCdiffs, SL_allvox, non_nan_mask)
    #     save_nii(savename + '.nii', header_fpath, vox3d)
    #     save_nii(savename + '_q.nii', header_fpath, qvals)

    if analysis_type == 2 or analysis_type == 5 or analysis_type == 7:
        vox3d, qvals = get_vox_map(sl_AUCdiffs, SL_allvox, non_nan_mask)
        for i in range(vox3d.shape[3]):
            save_nii(savename + '_' + str(i) + '.nii', header_fpath, vox3d[:,:,:,i])
            save_nii(savename + '_' + str(i) + '_q.nii', header_fpath, qvals[:,:,:,i])

        for sl_i in range(nSL):
            sl_AUCdiffs[sl_i] = sl_AUCdiffs[sl_i].mean(0)
        vox3d, qvals = get_vox_map(sl_AUCdiffs, SL_allvox, non_nan_mask)
        save_nii(savename + '_mean.nii', header_fpath, vox3d)
        save_nii(savename + '_mean_q.nii', header_fpath, qvals)

        if analysis_type == 5:
            coords = np.transpose(np.where(non_nan_mask))
            perm_maps = get_vox_map([sl[:,np.newaxis] for sl in sl_AUCdiffs], SL_allvox, non_nan_mask, return_q = False)
            vals = perm_maps[non_nan_mask]
            spear = np.zeros((nPerm, 3))
            for p in range(nPerm):
                spear[p,:] = spearmanr(vals[:,p], coords)[0][0,1:]
            print('Spearman corr ZYX=',spear[0,:])
            print('p vals=',norm.sf((spear[0,:]-spear[1:,:].mean(0))/np.std(spear[1:,:], axis=0)))
            
                    
    if analysis_type == 3:
        vox3d = get_vox_map(sl_K, SL_allvox, non_nan_mask, return_q=False)
        save_nii(savename + '_K.nii', header_fpath, vox3d)


    # if analysis_type == 4:
    #     vox3d_I, qvals_I = get_vox_map(sl_AUCdiffs_Intact, SL_allvox, non_nan_mask)
    #     vox3d_SF, qvals_SF = get_vox_map(sl_AUCdiffs_SFix, SL_allvox, non_nan_mask)
    #     save_nii(savename + '_Intact.nii', header_fpath, vox3d_I)
    #     save_nii(savename + '_Intact_q.nii', header_fpath, qvals_I)
    #     save_nii(savename + '_SFix.nii', header_fpath, vox3d_SF)
    #     save_nii(savename + '_SFix_q.nii', header_fpath, qvals_SF)

    if analysis_type == 6:
        vox3d_I, qvals_I = get_vox_map(sl_AUCdiffs_Intact, SL_allvox, non_nan_mask)
        vox3d_SF, qvals_SF = get_vox_map(sl_AUCdiffs_SFix, SL_allvox, non_nan_mask)
        for i in range(vox3d_I.shape[3]):
            save_nii(savename + '_Intact_' + str(i) + '.nii', header_fpath, vox3d_I[:,:,:,i])
            save_nii(savename + '_Intact_' + str(i) + '_q.nii', header_fpath, qvals_I[:,:,:,i])
            save_nii(savename + '_SFix_' + str(i) + '.nii', header_fpath, vox3d_SF[:,:,:,i])
            save_nii(savename + '_SFix_' + str(i) + '_q.nii', header_fpath, qvals_SF[:,:,:,i])
        
        sl_CondDiff = []
        for sl_i in range(nSL):
            sl_AUCdiffs_Intact[sl_i] = sl_AUCdiffs_Intact[sl_i].mean(0)
            sl_AUCdiffs_SFix[sl_i] = sl_AUCdiffs_SFix[sl_i].mean(0)
            sl_CondDiff.append(sl_AUCdiffs_Intact[sl_i]-sl_AUCdiffs_SFix[sl_i])
        vox3d_I, qvals_I = get_vox_map(sl_AUCdiffs_Intact, SL_allvox, non_nan_mask)
        vox3d_SF, qvals_SF = get_vox_map(sl_AUCdiffs_SFix, SL_allvox, non_nan_mask)
        vox3d_diff, qvals_diff = get_vox_map(sl_CondDiff, SL_allvox, non_nan_mask)
        save_nii(savename + '_Intact_mean.nii', header_fpath, vox3d_I)
        save_nii(savename + '_Intact_mean_q.nii', header_fpath, qvals_I)
        save_nii(savename + '_SFix_mean.nii', header_fpath, vox3d_SF)
        save_nii(savename + '_SFix_mean_q.nii', header_fpath, qvals_SF)
        save_nii(savename + '_diff_mean.nii', header_fpath, vox3d_diff)
        save_nii(savename + '_diff_mean_q.nii', header_fpath, qvals_diff)

    if analysis_type == 8:
        corrpeak = nSL * [None]
        for sl_i in range(nSL):
            corrpeak[sl_i] = np.zeros(nPerm)
            for p in range(nPerm):
                corrpeak[sl_i][p] = nearest_peak(sl_shift_corr[sl_i][:,p])

        corrpeak_3d, corrpeak_q = get_vox_map(corrpeak, SL_allvox, non_nan_mask)
        save_nii(savename + '.nii', header_fpath, corrpeak_3d)
        save_nii(savename + '_q.nii', header_fpath, corrpeak_q)



def get_vox_map(SL_results, SL_voxels, non_nan_mask, return_q=True, return_z=False):
    # SL_results is list of sl results, each length perm or size maps x perm
    """Projects searchlight results to voxel maps.

    Parameters
    ----------
    SL_results: list
        Results of the searchlight analysis from s_light function.

    SL_voxels: list
        Voxel information from searchlight analysis

    non_nan_mask: ndarray
        Voxel x voxel x voxel boolean mask indicating elements containing data

    Returns
    -------
    voxel_3dmap_rep1 : ndarray
        Repetition 1's 3d voxel map of results

    voxel_3dmap_lastreps : ndarray
        Last repetitions' 3d voxel map of results
    """

    coords = np.transpose(np.where(non_nan_mask))
    nVox = coords.shape[0]
    if SL_results[0].ndim == 1:
        nMaps = 1
        nPerm = len(SL_results[0])
    else:
        nMaps = SL_results[0].shape[0]
        nPerm = SL_results[0].shape[1]

    voxel_maps = np.zeros((nMaps, nPerm, nVox))
    voxel_SLcount = np.zeros(nVox)

    for idx, sl in enumerate(SL_voxels):
        if nMaps == 1:
            voxel_maps[0,:,sl] += SL_results[idx]
        else:
            for m in range(nMaps):
                voxel_maps[m,:,sl] += SL_results[idx][m, :]
        voxel_SLcount[sl] += 1

    nz_vox = voxel_SLcount > 0
    voxel_maps[:, :, nz_vox] = voxel_maps[:, :, nz_vox] / voxel_SLcount[nz_vox]
    voxel_maps[:, :, ~nz_vox] = np.nan

    vox3d = np.full(non_nan_mask.shape + (nMaps,), np.nan)
    vox3d[non_nan_mask,:] = voxel_maps[:,0,:].T

    if not return_q and not return_z:
        return vox3d.squeeze()

    null_means = voxel_maps[:, 1:, nz_vox].mean(1)
    null_stds = np.std(voxel_maps[:, 1:, nz_vox], axis=1)

    #!!
    # if np.any(null_stds == 0):
    #     const_vox = np.where(null_stds[0,:] == 0)[0]
    #     for v in const_vox:
    #         print(v, voxel_maps[0, 0, v], voxel_maps[0, 1, v])

    z = (voxel_maps[:, 0, nz_vox] - null_means)/null_stds
    p = norm.sf(z)
    q = np.zeros(p.shape)
    for m in range(nMaps):
        q[m,:] = FDR_p(p[m,:])

    z3d = np.full(non_nan_mask.shape + (nMaps,), np.nan)
    z3d[non_nan_mask,:] = z.T
    q3d = np.full(non_nan_mask.shape + (nMaps,), np.nan)
    q3d[non_nan_mask,:] = q.T

    if return_q and return_z:
        return vox3d.squeeze(), q3d.squeeze(), z3d.squeeze()
    elif return_q:
        return vox3d.squeeze(), q3d.squeeze()
    elif return_z:
        return vox3d.squeeze(), z3d.squeeze()

