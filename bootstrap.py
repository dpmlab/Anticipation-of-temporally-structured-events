import numpy as np
import random, os, fnmatch, multiprocessing as mp, math
from copy import deepcopy
from time import time

from utils.Dataset import Dataset, get_repdata, get_maskdata, get_non_nan_mask
import nibabel as nib

from s_light import s_light, get_vox_map
from analysis import get_AUCs, ev_annot_freq, hrf_convolution, get_DTs
from trial_jointfit import tj_fit
from scipy.stats import pearsonr


def bootstrap(n_resamp, data_fpath, mask, subj_regex, subjs=[], randomize=True, label='Intact', percent_cpu=0.75):

    # EXCEPTION HANDLING: do not let users use more than 90% of CPUs... #

    """

    Generate a bootstrap distribution of HMM fits on resampled data.

    Uses multiprocessing to complete bootstrap computations in parallel.

    :param n_resamp: int
        Number of times to resample original dataset for the bootstrap distribution.

    :param data_fpath: string
        Name of filepath pointing to the directory in which the data for bootstrapping is stored.

    :param mask: array_like
        Voxel x voxel x voxel boolean mask indicating elements that contain data, passed as non-nan mask parameter
        in the searchlight function.

    :param subj_regex: string
        Regular expression identifying all data file names.

    :param subjs: list, optional
        Option for manually indicating subjects to be considered for bootstrapping.

    :param randomize: boolean, optional
        Allows for resampling, with replacement, from the original dataset. Set to False if manually adding subjs.

    :param label: string, optional
        Used to identify subject files by condition (e.g. 'Intact' vs 'SFix' in GBH dataset).

    :param percent_cpu: float, optional
        Amount of CPUs to use in multiprocessing. 0.75 corresponds to 75%. Cannot exceed 90% CPU usage.

    :return resamp_aucs_rep1: ndarray
        Bootstrapped AUC (Area Under the Curve) results for the first repetition.

    :return resamp_aucs_last: ndarray
        Bootstrapped AUC results for the last repetitions.

    :return resamp_aucs_diff: ndarray
        Bootstrapped results of the last repetition's AUC minus first (resamp_aucs_last - resamp_aucs_last).

    :return resamped_subjs: list
        A list of the subjects used in each bootstrap.

    """

    if len(subjs) > 0:
        subjects = deepcopy(subjs)
    else:
        subjects = [subjdir for subjdir in os.listdir(data_fpath) if fnmatch.fnmatch(subjdir, subj_regex)]

    resamped_subjs = []
    results = []

    cpus = math.floor(mp.cpu_count() * percent_cpu)
    pool = mp.Pool(processes=cpus)

    time_start = time()

    for resamp in range(n_resamp):

        if randomize:
            resamp_subjs = [random.choice(subjects) for subject in range(len(subjects))]
            resamped_subjs.append(resamp_subjs)

        else:
            resamp_subjs = deepcopy(subjects[resamp])

        process = pool.apply(slight_aucs, args=(resamp_subjs, data_fpath, label, mask))
        results.append(process)

    time_end = time()

    print("total time for {} resamplings = {} minutes".format(n_resamp, round((time_end - time_start) / 60, 4)))


    resamp_aucs_rep1 = np.array(results)[:, 0, :, :, :]
    resamp_aucs_last = np.array(results)[:, 1, :, :, :]
    resamp_aucs_diff = np.array(results)[:, 2, :, :, :]

    return resamp_aucs_rep1, resamp_aucs_last, resamp_aucs_diff, resamped_subjs


def slight_aucs(subjects, data_fpath, label, mask_data, nevents=7, subj_regex='*pred*'):

    """

    Executes searchlight analysis on bootstrapped datasets, and computes AUCs for HMM fits.

    :param subjects: list
        List of subjects to include in the HMM fit.

    :param data_fpath: string
        Name of filepath pointing to the directory in which the data is stored.

    :param label: string
        Used to identify subject files by condition (e.g. 'Intact' vs 'SFix' in GBH dataset).

    :param mask_data: array_like
        Voxel x voxel x voxel boolean mask indicating elements that contain data, passed as non-nan mask parameter
        in the searchlight function.

    :param nevents: int, optional
        Specify number of boundaries to test.

    :param subj_regex: string, optional
        Regular expression identifying all data file names.

    :return vox3d_rep1: ndarray
        Repetition 1's 3d voxel map of AUC results

    :return vox3d_lastreps: ndarray
        Last repetitions' average's 3d voxel map of AUC results

    :return vox_AUCdiffs: ndarray
        AUC results of repetition 1 - last repetitions.


    """


    Resamp = Dataset(data_fpath)
    Resamp.data = get_repdata(Resamp.fpath, subj_regex, 'filt*' + label + '*', subjs=subjects)

    # get slight res
    sl_res, sl_vox = s_light(Resamp.data, mask_data)

    # get aucs and diffs
    sl_aucs = [get_AUCs(nevents, 2, segs) for segs in sl_res]

    vox3d_rep1, vox3d_lastreps = get_vox_map(sl_aucs, sl_vox, mask_data)
    vox_AUCdiffs = vox3d_lastreps - vox3d_rep1

    return vox3d_rep1, vox3d_lastreps, vox_AUCdiffs


def bootstrap_lagcorrs(n_resamp, dil_cluster_mask_fpath, cluster_mask_fpath,ev_annot, n_subjs=30):

    results = []

    cpus = math.floor(mp.cpu_count() * .75)
    pool = mp.Pool(processes=cpus)

    ev_annot_frequencies = ev_annot_freq(ev_annot)
    ev_annot_convolved = hrf_convolution(ev_annot_frequencies, 14)

    roi_dilated = np.asarray(nib.load(dil_cluster_mask_fpath).get_fdata())
    roi_clusters = np.asarray(nib.load(cluster_mask_fpath).get_fdata())

    n_rois = int(np.amax(roi_dilated))

    for resamp in range(n_resamp):

        resamp_subjs = np.random.randint(n_subjs, size=n_subjs)

        process = pool.apply(lag0_corr, args=(resamp_subjs, n_rois, roi_dilated, roi_clusters, ev_annot_convolved))
        results.append(process)

    return np.array(results)[:, 0, :, :, :], np.array(results)[:, 1, :, :, :], np.array(results)[:, 2, :, :, :], np.array(results)[:, 3, :, :, :]


def lag0_corr(subj_idxs, n_rois, roi_dilated, roi_clusters, ev_conv, vox_map_shape=(121, 145, 121)):

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

        segs = tj_fit(cluster_data)

        dts_first = get_DTs(segs[0])
        dts_last = get_DTs(segs[1])

        first_lag0corr[mask] = pearsonr(dts_first, ev_conv[1:])[0]
        lasts_lag0corr[mask] = pearsonr(dts_last, ev_conv[1:])[0]


        dil_clust_data = d_in

        dil_segs = tj_fit(dil_clust_data)

        dil_dts_first = get_DTs(dil_segs[0])
        dil_dts_lasts = get_DTs(dil_segs[1])

        dil_first_lag0corr[dilated_mask] = pearsonr(dil_dts_first, ev_conv[1:])[0]
        dil_lasts_lag0corr[dilated_mask] = pearsonr(dil_dts_lasts, ev_conv[1:])[0]

    return first_lag0corr, lasts_lag0corr, dil_first_lag0corr, dil_lasts_lag0corr