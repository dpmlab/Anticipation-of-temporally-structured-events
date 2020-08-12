import numpy as np
import random, os, fnmatch, multiprocessing as mp, math
from copy import deepcopy
from utils.pickles import arr_to_nii, arr_to_hdf5
from time import time
from datetime import date

from utils.Dataset import Dataset, get_repdata, get_maskdata, get_non_nan_mask
from utils.labels import get_label
import nibabel as nib

from s_light import s_light, get_vox_map
from analysis import get_AUCs, ev_annot_freq, hrf_convolution, get_DTs
from trial_jointfit import tj_fit
from scipy.stats import pearsonr


def bootstrap(n_resamp, condition, data_fpath, mask, subj_regex, subjs=[], randomize=False):

    if len(subjs) > 0:
        subjects = deepcopy(subjs)
    else:
        subjects = [subjdir for subjdir in os.listdir(data_fpath) if fnmatch.fnmatch(subjdir, subj_regex)]

    label = get_label(condition)

    resamped_subjs = []

    results = []

    cpus = math.floor(mp.cpu_count() * .75)
    pool = mp.Pool(processes=cpus)

    time_start = time()

    # for resamp in range(n_resamp):
    #     process = mp.Process(target=resample, args=(subjects, data_fpath, label, mask))
    #     processes.append(process)
    #     process.start()

    for resamp in range(n_resamp):
        if randomize:
            resamp_subjs = rand_subjs(subjects)
            resamped_subjs.append(resamp_subjs)
        else:
            resamp_subjs = deepcopy(subjects[resamp])
        process = pool.apply(resample, args=(resamp_subjs, data_fpath, label, mask))
        results.append(process)

    # results = [pool.apply(resample, args=(subjects, data_fpath, label, mask)) for resamp in n_resamp]

    # start processes #
    # for process in processes:
    #     process.start()

    # return processes #
    # for process in processes:
    #     process.join()

    # results = [output.get() for process in processes]

    time_end = time()

    print("total time for {} resamplings = {} minutes".format(n_resamp, round((time_end - time_start) / 60, 4)))

    # for sample in range(num_resamp):
    #
    #     vox3d_rep1, vox3d_lastreps, vox_AUCdiffs = resample(subjects, fpath, label, mask.data)
    #
    #     resamp_aucs_rep1.append(vox3d_rep1)
    #     resamp_aucs_last.append(vox3d_lastreps)
    #     resamp_aucs_diff.append(vox_AUCdiffs)

    # return results: range of diffs, is in conf int, etc.


    resamp_aucs_rep1 = np.array(results)[:, 0, :, :, :]
    resamp_aucs_last = np.array(results)[:, 1, :, :, :]
    resamp_aucs_diff = np.array(results)[:, 2, :, :, :]

    # arr_to_hdf5('subjs_bootstrap_' + condition + str(date.today()), np.array(results)[3])

    # arr_to_nii('avg_diffs_resamps.nii', header_file_fpath, np.nanmean(resamp_aucs_diff, axis=0))

    return resamp_aucs_rep1, resamp_aucs_last, resamp_aucs_diff, resamped_subjs


def rand_subjs(subjs):
    return [random.choice(subjs) for subj in range(len(subjs))]


def resample(subjects, data_fpath, label, mask_data, nevents=7, subj_regex='*pred*'):

    Resamp = Dataset(data_fpath)
    Resamp.data = get_repdata(Resamp.fpath, subj_regex, 'filt*' + label + '*', subjs=subjects)

    # get slight res
    sl_res, sl_vox = s_light(Resamp.data, mask_data, label)

    # get aucs and diffs
    sl_aucs = [get_AUCs(nevents, 2, segs) for segs in sl_res]

    vox3d_rep1, vox3d_lastreps = get_vox_map(sl_aucs, sl_vox, mask_data)
    vox_AUCdiffs = vox3d_lastreps - vox3d_rep1

    return vox3d_rep1, vox3d_lastreps, vox_AUCdiffs

def bootstrap_lagcorrs(n_resamp, condition, dil_cluster_mask_fpath, cluster_mask_fpath, n_subjs=30):

    # if len(subjs) > 0:
    #     subjects = deepcopy(subjs)
    # else:
    #     subjects = [subjdir for subjdir in os.listdir(data_fpath) if fnmatch.fnmatch(subjdir, subj_regex)]

    label = get_label(condition)

    results = []

    cpus = math.floor(mp.cpu_count() * .75)
    pool = mp.Pool(processes=cpus)

    # time_start = time()

    ev_annot = np.asarray(
        [5, 12, 54, 77, 90,
         3, 12, 23, 30, 36, 43, 50, 53, 78, 81, 87, 90,
         11, 23, 30, 50, 74,
         1, 55, 75, 90,
         4, 10, 53, 77, 82, 90,
         11, 54, 77, 81, 90,
         12, 22, 36, 54, 78,
         12, 52, 79, 90,
         10, 23, 30, 36, 43, 50, 77, 90,
         13, 55, 79, 90,
         4, 10, 23, 29, 35, 44, 51, 56, 77, 80, 85, 90,
         11, 55, 78, 90,
         11, 30, 43, 54, 77, 90,
         4, 11, 24, 30, 38, 44, 54, 77, 90]
    )

    ev_annot_frequencies = ev_annot_freq(ev_annot)
    ev_annot_convolved = hrf_convolution(ev_annot_frequencies, 14)

    roi_dilated = np.asarray(nib.load(dil_cluster_mask_fpath).get_fdata())
    roi_clusters = np.asarray(nib.load(cluster_mask_fpath).get_fdata())

    n_rois = int(np.amax(roi_dilated))

    for resamp in range(n_resamp):

        resamp_subjs = np.random.randint(n_subjs, size=n_subjs)

        process = pool.apply(lag0_corr, args=(resamp_subjs, label, n_rois, roi_dilated, roi_clusters, ev_annot_convolved))
        results.append(process)

    return np.array(results)[:, 0, :, :, :], np.array(results)[:, 1, :, :, :], np.array(results)[:, 2, :, :, :], np.array(results)[:, 3, :, :, :]


def lag0_corr(subj_idxs, label, n_rois, roi_dilated, roi_clusters, ev_conv, vox_map_shape=(121, 145, 121)):

    from utils.Dataset import get_roidata

    first_lag0corr = np.full(shape=vox_map_shape, fill_value=np.nan)
    lasts_lag0corr = np.full(shape=vox_map_shape, fill_value=np.nan)

    dil_first_lag0corr = np.full(shape=vox_map_shape, fill_value=np.nan)
    dil_lasts_lag0corr = np.full(shape=vox_map_shape, fill_value=np.nan)

    for cluster in range(1, n_rois + 1):

        d_in = get_roidata(roi=cluster, subj_idxs=subj_idxs)

        mask = np.logical_and(roi_dilated == cluster, roi_clusters)
        dilated_mask = roi_dilated == cluster

        roi_mask = mask.astype(int) + dilated_mask.astype(int)
        cluster_mask = roi_mask[np.where(roi_mask > 0)]
        cluster_mask = np.where(cluster_mask == 2, cluster_mask, 0)

        cluster_data = d_in[:, :, cluster_mask.astype(bool)]

        segs = tj_fit(label, cluster_data)

        dts_first = get_DTs(segs[0])
        dts_last = get_DTs(segs[1])

        first_lag0corr[mask] = pearsonr(dts_first, ev_conv[1:])[0]
        lasts_lag0corr[mask] = pearsonr(dts_last, ev_conv[1:])[0]


        dil_clust_data = d_in

        dil_segs = tj_fit(label, dil_clust_data)

        dil_dts_first = get_DTs(dil_segs[0])
        dil_dts_lasts = get_DTs(dil_segs[1])

        temp1 = pearsonr(dil_dts_first, ev_conv[1:])[0]
        temp2 = pearsonr(dil_dts_lasts, ev_conv[1:])[0]

        dil_first_lag0corr[dilated_mask] = pearsonr(dil_dts_first, ev_conv[1:])[0]
        dil_lasts_lag0corr[dilated_mask] = pearsonr(dil_dts_lasts, ev_conv[1:])[0]

    return first_lag0corr, lasts_lag0corr, dil_first_lag0corr, dil_lasts_lag0corr