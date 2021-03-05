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


def s_light(avg_lasts, tune_K, SRM_features, use_SFix,
            sl_path, subj_perms, non_nan_mask, header_fpath, savename):
    """Fits HMM to searchlights jointly to first and averaged last viewings.

    Executes searchlight analysis on voxel x voxel x voxel data, for all
    non_nan voxels. Stride and radius are used to adjust searchlight size and
    movement, and only searchlights with a number of valid voxels above a
    minimum size are run.

    Parameters
    ----------
    dataset : Dataset
        Data and mask for searchlight analysis
    stride : int, optional
        Specifies amount by which searchlights move across data
    radius : int, optional
        Specifies radius of each searchlight
    min_vox : int, optional
        Indicates the minimum number of elements with data for each searchlight

    Returns
    -------
    SL_results : list
        Results of HMM fits on searchlights
    SL_allvox : list
        Voxels in each searchlight
    """

    nSL = len(glob.glob(sl_path + '*.h5'))
    with open (sl_path + 'SL_allvox.p', 'rb') as fp:
        SL_allvox = pickle.load(fp)

    if not use_SFix:
        sl_AUCs = []
        sl_K = []
        for i in tqdm(range(nSL)):
            sl_h5 = tables.open_file(sl_path + str(i) + '.h5', mode='r')
            subj_list = []
            for subj in subj_perms:
                subjname = '/subj_' + subj.split('/')[-1]
                d = sl_h5.get_node(subjname, 'Intact').read()
                d = d[subj_perms[subj]]
                subj_list.append(d)
            sl_h5.close()

            if tune_K:
                K_range = np.arange(2, 10)
                ll = np.zeros(len(K_range))
                split = np.array([('predtrw01' in s) for s in subj_perms])
                rep1 = np.array([d[0] for d in subj_list])
                for i, K in enumerate(K_range):
                    ll[i] = heldout_ll(rep1, K, split)
                n_events = K_range[np.argmax(ll)]
                sl_K.append([n_events])
            else:
                n_events = 7

            if SRM_features == 0:
                group_data = np.mean(subj_list, axis=0)
            else:
                hyp_data = hyperalign(subj_list, nFeatures=SRM_features)
                group_data = np.mean(hyp_data, axis=0)

            seg = tj_fit(group_data, n_events=n_events, avg_lasts=avg_lasts)
            
            sl_AUCs.append(get_AUCs(seg))

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
            AUC_diff_Intact = []
            AUC_diff_SFix = []
            for group in ['predtrw01', 'predtrw02']:
                sl_h5 = tables.open_file(sl_path + str(i) + '.h5', mode='r')
                Intact_subj_list = []
                SFix_subj_list = []
                for subj in [s for s in subj_perms if group in s]:
                    subjname = '/subj_' + subj.split('/')[-1]
                    dI = sl_h5.get_node(subjname, 'Intact').read()
                    Intact_subj_list.append(dI)
                    dS = sl_h5.get_node(subjname, 'SFix').read()
                    SFix_subj_list.append(dS)
                sl_h5.close()

                group_Intact = np.mean(Intact_subj_list, axis=0)
                group_SFix = np.mean(SFix_subj_list, axis=0)

                AUC_Intact = get_AUCs(tj_fit(group_Intact, avg_lasts=avg_lasts))
                AUC_SFix = get_AUCs(tj_fit(group_SFix, avg_lasts=avg_lasts))

                AUC_diff_Intact.append(AUC_Intact - AUC_Intact[0])
                AUC_diff_SFix.append(AUC_SFix - AUC_SFix[0])
                
            sl_AUCdiffs_Intact.append(np.mean(AUC_diff_Intact, axis=0))
            sl_AUCdiffs_SFix.append(np.mean(AUC_diff_SFix, axis=0))

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

