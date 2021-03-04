from copy import deepcopy
import time
import numpy as np
from scipy.spatial.distance import cdist
import tables
import glob
import pickle
from tqdm import tqdm
from utils import get_AUCs, tj_fit, save_nii, hyperalign

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

def s_light_analysis(args):
    sl = args[0]
    subjects = args[1]

    sl_h5 = tables.open_file(sl, mode='r')
    subj_list = []
    for subj in subjects:
        subjname = '/subj_' + subj.split('/')[-1]
        d = sl_h5.get_node(subjname, 'Intact').read()
        subj_list.append(d)
    sl_h5.close()

    # Non-SRM version
    #group_data = np.mean(subj_list, axis=0) 
    #AUC = get_AUCs(tj_fit(group_data))

    # SRM version
    hyp_data = hyperalign(subj_list)
    group_data = np.mean(hyp_data, axis=0)
    AUC = get_AUCs(tj_fit(group_data))

    return AUC

def s_light(avg_lasts, SRM_features, sl_path, subjects, non_nan_mask, header_fpath, savename):
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

    sl_aucs = []
    for i in tqdm(range(nSL)):
        sl_h5 = tables.open_file(sl_path + str(i) + '.h5', mode='r')
        subj_list = []
        for subj in subjects:
            subjname = '/subj_' + subj.split('/')[-1]
            d = sl_h5.get_node(subjname, 'Intact').read()
            subj_list.append(d)
        sl_h5.close()

        if SRM_features == 0:
            group_data = np.mean(subj_list, axis=0)
        else:
            hyp_data = hyperalign(subj_list, nFeatures=SRM_features)
            group_data = np.mean(hyp_data, axis=0)
        
        sl_aucs.append(get_AUCs(tj_fit(group_data, avg_lasts=avg_lasts)))


    with open (sl_path + 'SL_allvox.p', 'rb') as fp:
        SL_allvox = pickle.load(fp)

    vox3d = get_vox_map(sl_aucs, SL_allvox, non_nan_mask)
    for i in range(1, vox3d.shape[3]):
        vox_AUCdiffs = vox3d[:,:,:,i] - vox3d[:,:,:,0]
        if vox3d.shape[3] == 2:
            save_nii(savename, header_fpath, vox_AUCdiffs)
        else:
            save_nii(savename + '_' + str(i) + '.nii', header_fpath, vox_AUCdiffs)


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

    voxel_maps /= voxel_SLcount

    voxel_3dmaps = np.full(non_nan_mask.shape + (nMaps,), np.nan)
    voxel_3dmaps[non_nan_mask,:] = voxel_maps.T

    return voxel_3dmaps

