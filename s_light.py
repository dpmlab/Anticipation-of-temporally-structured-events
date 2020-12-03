from copy import deepcopy
import time
import numpy as np
from scipy.spatial.distance import cdist
from utils import get_AUCs, tj_fit, save_nii


def s_light(dataset, stride=5, radius=5, min_vox=20):
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
    coords = np.transpose(np.where(dataset.non_nan_mask))
    d = deepcopy(dataset.data)
    d = d[:, :, dataset.non_nan_mask]

    SL_allvox = []

    sl_vox_start = time.time()

    for x in range(0, np.max(coords, axis=0)[0] + stride, stride):
        for y in range(0, np.max(coords, axis=0)[1] + stride, stride):
            for z in range(0, np.max(coords, axis=0)[2] + stride, stride):
                dists = cdist(coords, np.array([[x, y, z]]))[:, 0]
                SL_vox = np.where(dists <= radius)[0]
                if len(SL_vox) >= min_vox:
                    SL_allvox.append(SL_vox)

    res_start = time.time()

    print("Time in minutes to get searchlight voxels = ",
          round((res_start - sl_vox_start) / 60, 3))

    print("Running " + str(len(SL_allvox)) + " searchlights")
    SL_results = [tj_fit(d[:, :, sl]) for sl in SL_allvox]

    res_end = time.time()
    print("Time in minutes to get searchlight results = ",
          round((res_end - res_start) / 60, 3))

    return SL_results, SL_allvox


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

    voxel_map_rep1 = np.zeros(coords.shape[0])
    voxel_map_lastreps = np.zeros(coords.shape[0])
    voxel_SLcount = np.zeros(coords.shape[0])

    for idx, sl in enumerate(SL_voxels):
        voxel_map_rep1[sl] += SL_results[idx][0]
        voxel_map_lastreps[sl] += SL_results[idx][1]
        voxel_SLcount[sl] += 1

    voxel_map_rep1 /= voxel_SLcount
    voxel_map_lastreps /= voxel_SLcount

    voxel_3dmap_rep1 = np.full(non_nan_mask.shape, np.nan)
    voxel_3dmap_lastreps = np.full(non_nan_mask.shape, np.nan)

    for i, result in enumerate(voxel_map_rep1):

        if result == 0 and voxel_map_lastreps[i] == 0:
            continue

        voxel_3dmap_rep1[coords[i][0], coords[i][1], coords[i][2]] = result
        voxel_3dmap_lastreps[coords[i][0], coords[i][1], coords[i][2]] = \
            voxel_map_lastreps[i]

    return voxel_3dmap_rep1, voxel_3dmap_lastreps


def run_s_light_auc(dataset, savename, header_fpath):
    """Runs searchlight on dataset and save AUC difference.

    Parameters
    ----------
    dataset : Dataset
        Data and mask for searchlight analysis
    savename: string
        Filename to save nifti result to
    header_fpath : string
        File to use as a nifti template
    """

    sl_res, sl_vox = s_light(dataset)
    sl_aucs = [get_AUCs(segs) for segs in sl_res]
    vox3d_rep1, vox3d_lastreps = get_vox_map(sl_aucs, sl_vox,
                                             dataset.non_nan_mask)
    vox_AUCdiffs = vox3d_lastreps - vox3d_rep1

    save_nii(savename, header_fpath, vox_AUCdiffs)
