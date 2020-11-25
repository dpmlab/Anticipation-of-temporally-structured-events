import numpy as np
import os, fnmatch
from datetime import date

import nibabel as nib
from copy import deepcopy
from utils.pickles import arr_to_nii

from analysis import FDR_p, get_AUCs, get_DTs, ev_annot_freq, hrf_convolution, lag_correlation
from scipy.stats import pearsonr

from scipy.ndimage.morphology import binary_dilation
from utils.Dataset import Dataset, get_maskdata, get_non_nan_mask, get_repdata

from trial_jointfit import tj_fit
from s_light import s_light, get_vox_map, individual_sl_res
from bootstrap import bootstrap, bootstrap_lagcorrs


def intact_main_analysis(mask_fpath, subj_dirpath, header_file_fpath, save_nonnan_mask=False, nonnan_mask_fpath='int_ds_nonnanmask',
                         save_n_nonnan=False, n_nonnan_fpath='int_ds_notnan', save_zscored_DS=False, zscored_DS_fpath='int_ds_zs_'):

    """

    :param mask_fpath:
    :param subj_dirpath:
    :param header_file_fpath:
    :param save_nonnan_mask:
    :param nonnan_mask_fpath:
    :param save_n_nonnan:
    :param n_nonnan_fpath:
    :param save_zscored_DS:
    :param zscored_DS_fpath:
    :return:
    """

    mask = Dataset(mask_fpath)
    mask.data = get_maskdata(mask.fpath)

    ds = Dataset(subj_dirpath)
    ds.data = get_repdata(ds.fpath, '*pred*', 'filt*Intact*')


    ds.non_nan_mask = get_non_nan_mask(ds.num_not_nan, mask.data)

    if save_nonnan_mask:
        arr_to_nii(nonnan_mask_fpath + str(date.today()) + '.nii', header_file_fpath, np.asarray(ds.non_nan_mask).T)
        print('saved non-nan mask')
    if save_n_nonnan:
        arr_to_nii(n_nonnan_fpath + str(date.today()) + '.nii', header_file_fpath, np.asarray(ds.num_not_nan[0]).T)
        print('saved number of non-nan values per voxel')
    if save_zscored_DS:
        arr_to_nii(zscored_DS_fpath + str(date.today()) + '.nii', header_file_fpath, np.asarray(ds.data))
        print('saved z-scored dataset')

    sl_res, sl_vox = s_light(ds.data, ds.non_nan_mask)

    print("number of searchlights = ", len(sl_res))

    sl_aucs = [get_AUCs(7, 2, segs) for segs in sl_res]
    vox3d_rep1, vox3d_lastreps = get_vox_map(sl_aucs, sl_vox, ds.non_nan_mask)

    arr_to_nii('vox3d_rep1AUC_int_slight_' + str(date.today()) + '.nii', header_file_fpath, vox3d_rep1.T)
    arr_to_nii('vox3d_lastrepsAUC_int_slight_' + str(date.today()) + '.nii', header_file_fpath, vox3d_lastreps.T)

    vox_AUCdiffs = vox3d_lastreps - vox3d_rep1
    arr_to_nii('voxAUCdiffs_int_slight_' + str(date.today()) + '.nii', header_file_fpath, vox_AUCdiffs.T)


def run_bootstrap(mask_fpath, subj_dirpath, header_file_fpath, n_batches=10, n_bootstraps=10):

    """

    :param mask_fpath:
    :param subj_dirpath:
    :param header_file_fpath:
    :param n_batches:
    :param n_bootstraps:
    :return:
    """

    mask = get_maskdata(mask_fpath)

    for batch in range(n_batches):

        rep1, last, diff = bootstrap(n_bootstraps, subj_dirpath, mask, '*pred*')

        arr_to_nii('auc_resamps_rep1_' + '_batch' + str(n_batches + 1) + str(date.today()) + '.nii', header_file_fpath, rep1)
        arr_to_nii('auc_resamps_last_' + '_batch' + str(n_batches + 1) + str(date.today()) + '.nii', header_file_fpath, last)
        arr_to_nii('auc_resamps_diffs_' + '_batch' + str(n_batches + 1) + str(date.today()) + '.nii', header_file_fpath, diff)


def fdr_correction(pval_filepath):

    p_vals = np.asarray(nib.load(pval_filepath).get_fdata())

    idx_non_nan_pvals = np.where(~np.isnan(p_vals.ravel()))[0]
    fdr_pvals = FDR_p(np.take(p_vals.ravel(), idx_non_nan_pvals))

    fdr_3d = np.full(p_vals.ravel().shape, np.nan)
    fdr_3d.put(idx_non_nan_pvals, fdr_pvals)

    return np.reshape(fdr_3d, p_vals.shape)


def get_individual_sl_res(vox_x, vox_y, vox_z, zscored_ds_fpath='int_ds_zs_0325.nii', mask_fpath='int_ds_nonnanmask_0408.nii',
                          header_fpath='all_subjs/0411161_predtrw02/filtFuncMNI_Intact_Rep5.nii'):

    data = np.asarray(nib.load(zscored_ds_fpath).get_fdata())
    non_nan_mask = get_maskdata(mask_fpath)

    sl_voxels = [(vox_x, vox_y, vox_z)]

    for idx, voxels in enumerate(sl_voxels):

        sl_res, sl_vox = individual_sl_res(data, non_nan_mask, 'Intact', x=voxels[0], y=voxels[1], z=voxels[2])
        arr_to_nii('ind_slres_x' + str(voxels[0]) + '_y' + str(voxels[1]) + '_z' + str(voxels[2]), header_fpath, np.asarray(sl_res))


def lag_corr_analysis(zscored_ds_fpath='int_ds_zs_0325.nii', main_analysis_fpath='results/intact_analysis/auc_diffs_maskedby_sffdr.nii',
                      annot_dset_fpath='ev_annot.npy', header_fpath='all_subjs/0411161_predtrw02/filtFuncMNI_Intact_Rep5.nii',
                      vox_map_shape=(121, 145, 121), max_lags=20, save_res=False):

    ## generate lag correlations from all surviving searchlights ##

    from scipy.ndimage.measurements import label

    rois = nib.load(main_analysis_fpath).get_fdata()
    rois[np.isnan(rois)] = 0
    roi_clusters = label(rois, structure=np.ones((3, 3, 3)))

    d_in = nib.load(zscored_ds_fpath).get_fdata()

    aucs_first = np.full(shape=vox_map_shape, fill_value=np.nan)
    aucs_lasts = np.full(shape=vox_map_shape, fill_value=np.nan)

    max_lag_first = np.full(shape=vox_map_shape, fill_value=np.nan)
    max_lag_lasts = np.full(shape=vox_map_shape, fill_value=np.nan)

    ev_annot = np.load(annot_dset_fpath, allow_pickle=True)
    ev_annot_frequencies = ev_annot_freq(ev_annot)
    ev_annot_convolved = hrf_convolution(ev_annot_frequencies, 14)

    for cluster in range(1, roi_clusters[1] + 1):

        mask = roi_clusters[0] == cluster
        cluster_data = d_in[:, :, mask]

        segs = tj_fit(cluster_data)
        aucs = get_AUCs(7, 2, segs)

        aucs_first[mask] = aucs[0]
        aucs_lasts[mask] = aucs[1]

        dts_first = get_DTs(segs[0])
        dts_lasts = get_DTs(segs[1])

        max_lag_first[mask] = np.argmax(lag_correlation(x=dts_first, y=ev_annot_convolved[1:], max_lags=max_lags)) - max_lags
        max_lag_lasts[mask] = np.argmax(lag_correlation(x=dts_lasts, y=ev_annot_convolved[1:], max_lags=max_lags)) - max_lags

    if save_res:

        arr_to_nii('max_lag_TR_firsts.nii', header_fpath, max_lag_first)
        arr_to_nii('max_lag_TR_lasts.nii', header_fpath, max_lag_lasts)
        arr_to_nii('max_lag_TRs_first-lasts.nii', header_fpath, max_lag_first - max_lag_lasts)


def lag0_corr(main_analysis_fpath='results/intact_analysis/auc_diffs_maskedby_sffdr.nii', zscored_ds_fpath='int_ds_zs_0325.nii',
              annot_dset_fpath='ev_annot.npy', header_fpath='all_subjs/0411161_predtrw02/filtFuncMNI_Intact_Rep5.nii',
              vox_map_shape=(121, 145, 121), save_res=False):

    from scipy.ndimage.measurements import label

    rois = nib.load(main_analysis_fpath).get_fdata()

    roi_clusters = deepcopy(rois)
    roi_clusters[np.isnan(roi_clusters)] = 0
    roi_clusters = label(roi_clusters, structure=np.ones((3, 3, 3)))

    roi_dilated = binary_dilation(roi_clusters[0], structure=np.ones((5, 5, 5), dtype=bool)).astype(int)
    roi_dilated = label(roi_dilated, structure=np.ones((3, 3, 3)))

    arr_to_nii('dilated_clusters.nii', header_fpath, roi_dilated[0])

    d_in = nib.load(zscored_ds_fpath).get_fdata()

    ev_annot = np.load(annot_dset_fpath, allow_pickle=True)

    ev_annot_frequencies = ev_annot_freq(ev_annot)
    ev_annot_convolved = hrf_convolution(ev_annot_frequencies, 14)

    dil_first_lag0corr = np.full(shape=vox_map_shape, fill_value=np.nan)
    dil_lasts_lag0corr = np.full(shape=vox_map_shape, fill_value=np.nan)

    first_lag0corr = np.full(shape=vox_map_shape, fill_value=np.nan)
    lasts_lag0corr = np.full(shape=vox_map_shape, fill_value=np.nan)

    for cluster in range(1, roi_dilated[1] + 1):

        mask = np.logical_and(roi_dilated[0] == cluster, roi_clusters[0])
        cluster_data = d_in[:, :, mask]

        segs = tj_fit(cluster_data)

        dts_first = get_DTs(segs[0])
        dts_lasts = get_DTs(segs[1])

        first_lag0corr[mask] = pearsonr(dts_first, ev_annot_convolved[1:])[0]
        lasts_lag0corr[mask] = pearsonr(dts_lasts, ev_annot_convolved[1:])[0]

        dil_mask = roi_dilated[0] == cluster
        dil_clust_data = d_in[:, :, dil_mask]

        dil_segs = tj_fit(dil_clust_data)

        dil_dts_first = get_DTs(dil_segs[0])
        dil_dts_lasts = get_DTs(dil_segs[1])

        dil_first_lag0corr[dil_mask] = pearsonr(dil_dts_first, ev_annot_convolved[1:])[0]
        dil_lasts_lag0corr[dil_mask] = pearsonr(dil_dts_lasts, ev_annot_convolved[1:])[0]

    if save_res:

        arr_to_nii('lag0corrs_first_original.nii', header_fpath, first_lag0corr)
        arr_to_nii('lag0corrs_lasts_original.nii', header_fpath, lasts_lag0corr)
        arr_to_nii('dilroi_lag0corrs_first_original.nii', header_fpath, dil_first_lag0corr)
        arr_to_nii('dilroi_lag0corrs_lasts_original.nii', header_fpath, dil_lasts_lag0corr)
        arr_to_nii('roi_clusters_mask.nii', header_fpath, roi_clusters[0])


def lag0_diffs_bstraps(first_bstraps_regex, lasts_bstraps_regex, fname_pos, fname_neg, res_dir='results/',
                       header_fpath='all_subjs/0411161_predtrw02/filtFuncMNI_Intact_Rep5.nii', mask_fpath='MNI152_T1_brain_resample.nii'):

    from analysis import get_percent_pos_neg

    first_resamp_files = [file for file in os.listdir(res_dir) if fnmatch.fnmatch(file, first_bstraps_regex)]
    first_resamps = np.concatenate([np.asarray(nib.load(res_dir + file).get_fdata()) for file in first_resamp_files], axis=0)

    lasts_resamp_files = [file for file in os.listdir(res_dir) if fnmatch.fnmatch(file, lasts_bstraps_regex)]
    lasts_resamps = np.concatenate([np.asarray(nib.load(res_dir + file).get_fdata()) for file in lasts_resamp_files], axis=0)

    diff_resamps = lasts_resamps - first_resamps

    mask_fpath = mask_fpath
    mask = get_maskdata(mask_fpath)
    mask = mask.T

    percent_pos = np.full(mask.shape, np.nan)
    percent_neg = np.full(mask.shape, np.nan)

    for x in range(0, mask.shape[0]):
        for y in range(0, mask.shape[1]):
            for z in range(0, mask.shape[2]):

                if mask[x, y, z] == 0:
                    continue

                percent_pos[x, y, z], percent_neg[x, y, z] = get_percent_pos_neg(diff_resamps[:, x, y, z])

    arr_to_nii(fname_pos, header_fpath, percent_pos)
    arr_to_nii(fname_neg, header_fpath, percent_neg)


def lag0_bootstrap(n_iter, header_fpath='all_subjs/0411161_predtrw02/filtFuncMNI_Intact_Rep5.nii', dil_cluster_fpath='dilated_clusters.nii',
                   cluster_fpath='roi_clusters_mask.nii'):

    # before running on server: change filepaths, change # of iterations, AND CHANGE # of SUBJECTS IN BOOTSTRAP #

    dil_cluster_mask = nib.load(dil_cluster_fpath).get_fdata()
    cluster_mask = nib.load(cluster_fpath).get_fdata()

    for iter in range(n_iter):

        first_lag0corr, lasts_lag0corr, dil_first_lag0corr, dil_lasts_lag0corr = bootstrap_lagcorrs(10, dil_cluster_mask, cluster_mask)

        arr_to_nii('firstlag0_batch' + str(iter + 1) + '_' + str(date.today()) + '.nii', header_fpath, first_lag0corr)
        arr_to_nii('lastslag0_batch' + str(iter + 1) + '_' + str(date.today()) + '.nii', header_fpath, lasts_lag0corr)
        arr_to_nii('dil_firstlag0_batch' + str(iter + 1) + '_' + str(date.today()) + '.nii', header_fpath, dil_first_lag0corr)
        arr_to_nii('dil_lastslag0_batch' + str(iter + 1) + '_' + str(date.today()) + '.nii', header_fpath, dil_lasts_lag0corr)

def save_bstrap_clusters(res_dir, bstraps_regex, x, y, z, bstrap_type='', header_fpath='all_subjs/0411161_predtrw02/filtFuncMNI_Intact_Rep5.nii'):

    bstrap_files = [file for file in os.listdir(res_dir) if fnmatch.fnmatch(file, bstraps_regex)]
    bstraps = np.concatenate([np.asarray(nib.load(res_dir + file).get_fdata()) for file in bstrap_files], axis=0)

    arr_to_nii('bstrap_res_x=' + str(x) + 'y=' + str(y) + 'z=' + str(z) + bstrap_type, header_fpath, np.asarray(bstraps[:, x, y, z]))

