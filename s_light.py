from copy import deepcopy
import numpy as np
from scipy.spatial.distance import cdist
import time
from trial_jointfit import tj_fit
from utils.pickles import arr_to_hdf5


def s_light(data, non_nan_mask, condition, stride=5, radius=5, min_vox=20, save_res=False):


    coords = np.transpose(np.where(non_nan_mask))
    d = np.asarray(deepcopy(data))
    d = d[:, :, non_nan_mask]

    SL_allvox = []

    # debugging 02/2018

    # distances = cdist(coords, np.array([86, 123, 49]).reshape((1,3)))[:, 0]
    # #distances = cdist(coords, np.array([86, 112, 53]).reshape((1, 3)))[:, 0]
    # sorted = np.argsort(distances, kind='mergesort')
    # SL_vox = np.where(distances <= radius)[0]
    # print(len(SL_vox))
    # if len(SL_vox) >= min_vox:
    #     SL_allvox.append(SL_vox)
    #     #SL_coords.append(np.array([88, 114, 53]))

    # end debugging 02/2018

    sl_vox_start = time.time()

    for x in range(0, np.max(coords, axis=0)[0] + stride, stride):
        for y in range(0, np.max(coords, axis=0)[1] + stride, stride):
            for z in range(0, np.max(coords, axis=0)[2] + stride, stride):


                distances = cdist(coords, np.array([x,y,z]).reshape((1, 3)))[:, 0]
                SL_vox = np.where(distances <= radius)[0]
                if len(SL_vox) >= min_vox:
                    SL_allvox.append(SL_vox)

    res_start = time.time()

    print("time in minutes to get sl voxels = ", round((res_start - sl_vox_start) / 60, 3))

    SL_results = [tj_fit(condition, d[:, :, sl]) for sl in SL_allvox]

    if save_res:
        arr_to_hdf5('sl_res_' + condition + '.h5', SL_results)
        arr_to_hdf5('sl_allvox_' + condition + '.h5', SL_allvox)

    res_end = time.time()
    print("time in minutes to get sl results = ", round((res_end - res_start) / 60, 3))

    return SL_results, SL_allvox


def get_vox_map(SL_results, SL_voxels, non_nan_mask):

    coords = np.transpose(np.where(non_nan_mask))

    # coords = deepcopy(mask)

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

    for idx, result in enumerate(voxel_map_rep1):

        if result == 0 and voxel_map_lastreps[idx] == 0:
            continue

        voxel_3dmap_rep1[coords[idx][0], coords[idx][1], coords[idx][2]] = result
        voxel_3dmap_lastreps[coords[idx][0], coords[idx][1], coords[idx][2]] = voxel_map_lastreps[idx]

    return voxel_3dmap_rep1, voxel_3dmap_lastreps


def individual_sl_res(data, non_nan_mask, condition, x, y, z, radius=5, min_vox=20):

    coords = np.transpose(np.where(non_nan_mask))
    d = np.asarray(deepcopy(data))
    d = d[:, :, non_nan_mask]

    SL_allvox = []

    distances = cdist(coords, np.array([x, y, z]).reshape((1,3)))[:, 0]
    SL_vox = np.where(distances <= radius)[0]

    if len(SL_vox) >= min_vox:
        SL_allvox.append(SL_vox)

    SL_results = [tj_fit(condition, d[:, :, sl]) for sl in SL_allvox]

    return SL_results, SL_allvox