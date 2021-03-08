from copy import deepcopy
import time
import numpy as np
from scipy.spatial.distance import cdist
import tables
import glob
import pickle
from tqdm import tqdm
from utils import get_AUCs, tj_fit, save_nii, hyperalign, heldout_ll

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

def one_sl(subj_list, subjects, avg_lasts, tune_K, SRM_features):
    if tune_K:
        K_range = np.arange(2, 10)
        ll = np.zeros(len(K_range))
        split = np.array([('predtrw01' in s) for s in subjects])
        rep1 = np.array([d[0] for d in subj_list])
        for i, K in enumerate(K_range):
            ll[i] = heldout_ll(rep1, K, split)
        n_events = K_range[np.argmax(ll)]
    else:
        n_events = 7

    if SRM_features == 0:
        group_data = np.mean(subj_list, axis=0)
    else:
        hyp_data = hyperalign(subj_list, nFeatures=SRM_features)
        group_data = np.mean(hyp_data, axis=0)

    seg = tj_fit(group_data, n_events=n_events, avg_lasts=avg_lasts)
    
    return (get_AUCs(seg), n_events)

def one_sl_SF(group_Intact, group_SFix, avg_lasts):
    AUC_diff_Intact = []
    AUC_diff_SFix = []
    for group in range(2):
        AUC_Intact = get_AUCs(tj_fit(group_Intact[group], avg_lasts=avg_lasts))
        AUC_SFix = get_AUCs(tj_fit(group_SFix[group], avg_lasts=avg_lasts))

        AUC_diff_Intact.append(AUC_Intact - AUC_Intact[0])
        AUC_diff_SFix.append(AUC_SFix - AUC_SFix[0])

    return (np.mean(AUC_diff_Intact, axis=0),
            np.mean(AUC_diff_SFix, axis=0))

def load_pickle(analysis_type, perm_i, pickle_path):
    nSL = 5354
    sl_AUCs = nSL*[None]
    sl_K = nSL*[None]
    sl_AUCdiffs_Intact = nSL*[None]
    sl_AUCdiffs_SFix = nSL*[None]

    pickles = glob.glob(pickle_path + '*.p')
    for i, pick in enumerate(pickles):
        pick_parts = pick.split('_')
        analysis_i = int(pick_parts[1])
        sl_i = int(pick_parts[2])
        perm_start = int(pick_parts[3])
        perm_end = int(pick_parts[4])

        if analysis_i == analysis_type and perm_i >= perm_start and perm_i < perm_end:
            with open (pick, 'rb') as fp:
                pick_data = pickle.load(fp)
            perm_data = pick_data[perm_i - perm_start]
            if analysis_type == 4:
                sl_AUCdiffs_Intact[sl_i] = perm_data[0]
                sl_AUCdiffs_SFix[sl_i] = perm_data[1]
            elif analysis_type == 3:
                sl_AUCs[sl_i] = perm_data[0]
                sl_K[sl_i] = perm_data[1]
            else:
                sl_AUCs[sl_i] = perm_data

    AUC_Nones = np.sum([i is None for i in sl_AUCs])
    AUC_SFix_Nones = np.sum([i is None for i in sl_AUCdiffs_SFix])
    if analysis_type == 4:
        if AUC_SFix_Nones > 0:
            print('Missing', AUC_SFix_Nones, 'searchlights')
    else:
        if AUC_Nones > 0:
            print('Missing', AUC_Nones, 'searchlights')
    return (sl_AUCs, sl_K, sl_AUCdiffs_Intact, sl_AUCdiffs_SFix)

def compile_s_light(analysis_type, non_nan_mask, sl_path, header_fpath, pickle_path, savename):

    nSL = len(glob.glob(sl_path + '*.h5'))
    with open (sl_path + 'SL_allvox.p', 'rb') as fp:
        SL_allvox = pickle.load(fp)

    sl_AUCs = load_pickle(0, 0, pickle_path)[0]
    for i in range(len(sl_AUCs)):
        if sl_AUCs[i] is None:
            sl_AUCs[i] = [0, 0]
    vox3d = get_vox_map(sl_AUCs, SL_allvox, non_nan_mask)
    for i in range(1, vox3d.shape[3]):
        vox_AUCdiffs = vox3d[:,:,:,i] - vox3d[:,:,:,0]
        if vox3d.shape[3] == 2:
            save_nii(savename, header_fpath, vox_AUCdiffs)

    return

    if not use_SFix:
        for perm_i in range(100):
            all_sl = load_pickle(analysis_type, perm_i, pickle_path)

        vox3d = get_vox_map(sl_AUCs, SL_allvox, non_nan_mask)
        for i in range(1, vox3d.shape[3]):
            vox_AUCdiffs = vox3d[:,:,:,i] - vox3d[:,:,:,0]
            if vox3d.shape[3] == 2:
                save_nii(savename, header_fpath, vox_AUCdiffs)
            else:
                save_nii(savename + '_' + str(i) + '.nii', header_fpath, vox_AUCdiffs)
                if i == 1:
                    mean_AUCdiff = vox3d[:,:,:,1:].mean(3) - vox3d[:,:,:,0]
                    save_nii(savename + '_mean.nii', header_fpath, mean_AUCdiff)

        if tune_K:
            vox3d = get_vox_map(sl_K, SL_allvox, non_nan_mask)
            save_nii(savename[:-4] + '_K.nii', header_fpath, vox3d[:,:,:,0])
    else:
        sl_AUCdiffs_Intact = []
        sl_AUCdiffs_SFix = []
        for i in tqdm(range(nSL)):
            AUC_diff = one_sl_SF(sl_path + str(i) + '.h5', subj_perms, avg_lasts)
            sl_AUCdiffs_Intact.append(AUC_diff[0])
            sl_AUCdiffs_SFix.append(AUC_diff[1])


        vox3d_Intact = get_vox_map(sl_AUCdiffs_Intact, SL_allvox, non_nan_mask)
        vox3d_SFix = get_vox_map(sl_AUCdiffs_SFix, SL_allvox, non_nan_mask)
        for i in range(1, vox3d_Intact.shape[3]):
            if vox3d_Intact.shape[3] == 2:
                save_nii(savename + '_Intact.nii', header_fpath, vox3d_Intact[:,:,:,i])
                save_nii(savename + '_SFix.nii', header_fpath, vox3d_SFix[:,:,:,i])
            else:
                save_nii(savename + '_Intact_' + str(i) + '.nii', header_fpath, vox3d_Intact[:,:,:,i])
                save_nii(savename + '_SFix_' + str(i) + '.nii', header_fpath, vox3d_SFix[:,:,:,i])


def get_vox_map(SL_results, SL_voxels, non_nan_mask):

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
    nMaps = len(SL_results[0])
    nVox = coords.shape[0]

    voxel_maps = np.zeros((nMaps, nVox))
    voxel_SLcount = np.zeros(nVox)

    for idx, sl in enumerate(SL_voxels):
        for m in range(nMaps):
            voxel_maps[m,sl] += SL_results[idx][m]
        voxel_SLcount[sl] += 1

    nz_vox = voxel_SLcount > 0
    voxel_maps[:, nz_vox] = voxel_maps[:, nz_vox] / voxel_SLcount[nz_vox]
    voxel_maps[:, ~nz_vox] = np.nan

    voxel_3dmaps = np.full(non_nan_mask.shape + (nMaps,), np.nan)
    voxel_3dmaps[non_nan_mask,:] = voxel_maps.T

    return voxel_3dmaps

