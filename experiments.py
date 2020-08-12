import numpy as np
import os, fnmatch, time
from datetime import date

import nibabel as nib
from copy import deepcopy
from utils.pickles import arr_to_nii
from utils.Dataset import Dataset, get_maskdata, get_non_nan_mask, get_repdata, get_data_pval

from analysis import get_conf_int, get_pdf, get_sf, get_CV_sf, get_ks_test, FDR_p, get_AUCs, get_DTs, ev_annot_freq, hrf_convolution, lag_correlation, get_confval
from scipy.stats import pearsonr
from scipy.ndimage.morphology import binary_dilation

from s_light import s_light, get_vox_map, individual_sl_res
from bootstrap import bootstrap, bootstrap_lagcorrs
from trial_jointfit import tj_fit



def conf_int_bootstrap_res(orig_auc_fpath, bstraps_regex, res_dir='results/', mask_fpath='MNI152_T1_brain_resample.nii', is_lag0=True):

    resamp_files = [file for file in os.listdir(res_dir) if fnmatch.fnmatch(file, bstraps_regex)]
    resamp_aucs = np.concatenate([np.asarray(nib.load(res_dir + file).get_fdata()) for file in resamp_files], axis=0)

    original_aucs = np.asarray(nib.load(orig_auc_fpath).get_fdata()).T

    mask_fpath = mask_fpath
    mask = get_maskdata(mask_fpath)
    if is_lag0:
        mask = mask.T

    conf_int_min, conf_int_max, in_conf_int, n_less0, conf_min_val = np.full(mask.shape, np.nan), np.full(mask.shape, np.nan), np.full(mask.shape, np.nan), np.full(mask.shape, np.nan), np.full(mask.shape, np.nan)
    conf_mean, conf_std, conf_var, conf_pval, samp5 = np.full(mask.shape, np.nan), np.full(mask.shape, np.nan), np.full(mask.shape, np.nan), np.full(mask.shape, np.nan), np.full(mask.shape, np.nan)


    n_vox_nans = 0

    for x in range(0, mask.shape[0]):
        for y in range(0, mask.shape[1]):
            for z in range(0, mask.shape[2]):

                # if mask[x, y, z] == 0 or np.isnan(original_aucs[x, y, z]):
                if mask[x, y, z] == 0:
                    continue

                if np.count_nonzero(np.isnan(resamp_aucs[:, x, y, z])) >= 1:
                    n_vox_nans += 1

                conf_int_res = get_conf_int(original_aucs[x, y, z], resamp_aucs[:, x, y, z])

                in_conf_int[x, y, z], conf_int_min[x, y, z], conf_int_max[x, y, z] = conf_int_res[0], conf_int_res[1][0], conf_int_res[1][1]
                n_less0[x, y, z], conf_min_val[x, y, z] = conf_int_res[2], conf_int_res[3]

                conf_mean[x, y, z], conf_std[x, y, z], conf_var[x, y, z], conf_pval[x, y, z] = conf_int_res[4], conf_int_res[5], conf_int_res[6], conf_int_res[7]

                samp5[x, y, z] = conf_int_res[8]

    return in_conf_int, conf_int_min, conf_int_max, n_less0, conf_min_val, conf_mean, conf_std, conf_var, conf_pval, samp5


def get_conf_int_results():

    header_file_fpath = 'all_subjs/0411161_predtrw02/filtFuncMNI_Intact_Rep5.nii'
    #
    # in_conf_int, conf_int_min, conf_int_max, n_less0, conf_min_val, conf_mean, conf_std, conf_var, conf_pval, samp5 = conf_int_bootstrap_res('lag0corrs_first_original.nii', 'firstlag0*')
    # arr_to_nii('lag0_first_5th_sample.nii', header_file_fpath, samp5)

    # in_conf_int, conf_int_min, conf_int_max, n_less0, conf_min_val, conf_mean, conf_std, conf_var, conf_pval, samp5 = conf_int_bootstrap_res('lag0corrs_lasts_original.nii', 'lastslag0*')
    # arr_to_nii('lag0_lasts_5th_sample.nii', header_file_fpath, samp5)

    # in_conf_int, conf_int_min, conf_int_max, n_less0, conf_min_val, conf_mean, conf_std, conf_var, conf_pval, samp5 = conf_int_bootstrap_res('dilroi_lag0corrs_first_original.nii', 'dil_firstlag0*')
    # arr_to_nii('dillag0_first_5th_sample.nii', header_file_fpath, samp5)

    # in_conf_int, conf_int_min, conf_int_max, n_less0, conf_min_val, conf_mean, conf_std, conf_var, conf_pval, samp5 = conf_int_bootstrap_res('dilroi_lag0corrs_lasts_original.nii', 'dil_lastslag0*')
    # arr_to_nii('dillag0_lasts_5th_sample.nii', header_file_fpath, samp5)

def lag0_diffs_bstraps(first_bstraps_regex, lasts_bstraps_regex, fname_pos, fname_neg, res_dir='results/', mask_fpath='MNI152_T1_brain_resample.nii'):

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

    header_file_fpath = 'all_subjs/0411161_predtrw02/filtFuncMNI_Intact_Rep5.nii'
    arr_to_nii(fname_pos, header_file_fpath, percent_pos)
    arr_to_nii(fname_neg, header_file_fpath, percent_neg)




def pdf_sf_bootstrap_res(orig_auc_fpath, bstraps_regex, res_dir='results/', mask_fpath='MNI152_T1_brain_resample.nii'):

    resamp_files = [file for file in os.listdir(res_dir) if fnmatch.fnmatch(file, bstraps_regex)]
    resamp_aucs = np.concatenate([np.asarray(nib.load(res_dir + file).get_fdata()) for file in resamp_files], axis=0)

    original_aucs = np.asarray(nib.load(orig_auc_fpath).get_fdata()).T

    mask_fpath = mask_fpath
    mask = get_maskdata(mask_fpath)

    pdf_res, sf_res, cv_sf_res = np.full(mask.shape, np.nan), np.full(mask.shape, np.nan), np.full(mask.shape, np.nan)

    for x in range(0, mask.shape[0]):
        for y in range(0, mask.shape[1]):
            for z in range(0, mask.shape[2]):

                if mask[x, y, z] == 0 or np.isnan(original_aucs[x, y, z]):
                    continue

                resamp_mean = np.nanmean(resamp_aucs[:, x, y, z])
                resamp_std = np.nanstd(resamp_aucs[:, x, y, z])

                pdf_res[x, y, z], sf_res[x, y, z], cv_sf_res[x, y, z] = get_pdf(original_aucs[x, y, z], resamp_mean, resamp_std), get_sf(original_aucs[x, y, z], resamp_mean, resamp_std), get_CV_sf(resamp_mean, resamp_std)

    return pdf_res, sf_res, cv_sf_res


def ks_norm(orig_auc_fpath, bstraps_regex, res_dir='results/', mask_fpath='MNI152_T1_brain_resample.nii'):

    resamp_files = [file for file in os.listdir(res_dir) if fnmatch.fnmatch(file, bstraps_regex)]
    resamp_aucs = np.concatenate([np.asarray(nib.load(res_dir + file).get_fdata()) for file in resamp_files], axis=0)

    original_aucs = np.asarray(nib.load(orig_auc_fpath).get_fdata()).T

    mask_fpath = mask_fpath
    mask = get_maskdata(mask_fpath)

    ks_res = np.full(mask.shape, np.nan)

    for x in range(0, mask.shape[0]):
        for y in range(0, mask.shape[1]):
            for z in range(0, mask.shape[2]):

                if mask[x, y, z] == 0 or np.isnan(original_aucs[x, y, z]):
                    continue
                ks_res[x, y, z] = get_ks_test(resamp_aucs[:, x, y, z])

    return ks_res


def run_bootstrap():

    # change mask_fpath to point to intersection mask ###
    # mask_fpath = 'int_ds_nonnanmask_0408.nii'
    # mask = get_maskdata(mask_fpath)
    # subj_dirpath = '../../../data/gbh/data/'
    # header_file_fpath = '../../../data/gbh/data/0408161_predtrw02/filtFuncMNI_Intact_Rep1.nii'

    mask_fpath = 'MNI152_T1_brain_resample.nii'
    mask = get_maskdata(mask_fpath)
    subj_dirpath = 'all_subjs/'
    header_file_fpath = 'all_subjs/0408161_predtrw02/filtFuncMNI_Intact_Rep1.nii'

    for bs in range(10):

        rep1, last, diff = bootstrap(10, 'Intact', subj_dirpath, mask, '*pred*')

        arr_to_nii('auc_100resamps_rep1_' + str(date.today()) + '_batch' + str(bs + 1) + '.nii', header_file_fpath, rep1)
        arr_to_nii('auc_100resamps_last_' + str(date.today()) + '_batch' + str(bs + 1) + '.nii', header_file_fpath, last)
        arr_to_nii('auc_100resamps_diffs_' + str(date.today()) + '_batch' + str(bs + 1) + '.nii', header_file_fpath, diff)


def slight_saved_ds():

    mask_fpath = 'MNI152_T1_brain_resample.nii'
    subj_dirpath = 'all_subjs/'
    header_file_fpath = 'all_subjs/0408161_predtrw02/filtFuncMNI_Intact_Rep1.nii'

    mask = Dataset(mask_fpath)
    mask.data = get_maskdata(mask.fpath)

    ds = Dataset('orig_ds_zs.nii')
    ds.data = nib.load('orig_ds_zs.nii').get_fdata()
    ds.num_not_nan = nib.load('org_ds_notnan_rep1_0301.nii').get_fdata()
    ds.non_nan_mask = get_non_nan_mask(ds.num_not_nan, mask.data)

    sl_res, sl_vox = s_light(ds.data, ds.non_nan_mask, 'Intact')

    sl_aucs = [get_AUCs(7, 2, segs) for segs in sl_res]

    vox3d_rep1, vox3d_lastreps = get_vox_map(sl_aucs, sl_vox, ds.non_nan_mask)
    #
    vox_AUCdiffs = vox3d_lastreps - vox3d_rep1
    #
    arr_to_nii('vox3d_rep1AUC_orig_slight022818.nii', header_file_fpath, vox3d_rep1)
    arr_to_nii('vox3d_lastrepsAUC_orig_slight022818.nii', header_file_fpath, vox3d_lastreps)
    arr_to_nii('voxAUCdiffs_orig_slight022818.nii', header_file_fpath, vox_AUCdiffs)


def get_fdr(filepath):

    p_vals = np.asarray(nib.load(filepath).get_fdata())

    idx_non_nan_pvals = np.where(~np.isnan(p_vals.ravel()))[0]
    fdr_pvals = FDR_p(np.take(p_vals.ravel(), idx_non_nan_pvals))

    fdr_3d = np.full(p_vals.ravel().shape, np.nan)
    fdr_3d.put(idx_non_nan_pvals, fdr_pvals)

    return np.reshape(fdr_3d, p_vals.shape)


def get_fdr_sf_maskedvals(orig_aucs_fpath, sf_vals_fpath, p95_vals_fpath, sf_outpath, p95_outpath, sf_mask_outpath, p95_mask_outpath, header_fpath):

    header_file_fpath = header_fpath

    sf_fdr = get_fdr(sf_vals_fpath)
    arr_to_nii(sf_outpath, header_file_fpath, sf_fdr)

    p95_fdr = get_fdr(p95_vals_fpath)
    arr_to_nii(p95_outpath, header_file_fpath, p95_fdr)
    original_aucs = np.asarray(nib.load(orig_aucs_fpath).get_fdata())

    sf_fdr_maskedbyp05 = get_data_pval(original_aucs, sf_fdr)
    sf_fdr_maskedbyp05[sf_fdr_maskedbyp05 == 0] = np.nan
    p95_fdr_maskedbyp05 = get_data_pval(original_aucs, p95_fdr)
    p95_fdr_maskedbyp05[p95_fdr_maskedbyp05 == 0] = np.nan

    arr_to_nii(sf_mask_outpath, header_file_fpath, sf_fdr_maskedbyp05)
    arr_to_nii(p95_mask_outpath, header_file_fpath, p95_fdr_maskedbyp05)


def get_individual_sl_res(vox_x, vox_y, vox_z):
    data = np.asarray(nib.load('int_ds_zs_0325.nii').get_fdata())
    mask = 'int_ds_nonnanmask_0408.nii'
    non_nan_mask = get_maskdata(mask)
    header_file_fpath = 'all_subjs/0408161_predtrw02/filtFuncMNI_Intact_Rep2.nii'

    # sl_voxels = [(93, 38, 68), (44, 28, 56), (40, 114, 38)]
    sl_voxels = [(vox_x, vox_y, vox_z)]

    for idx, voxels in enumerate(sl_voxels):

        sl_res, sl_vox = individual_sl_res(data, non_nan_mask, 'Intact', x=voxels[0], y=voxels[1], z=voxels[2])
        arr_to_nii('ind_slres_x' + str(voxels[0]) + '_y' + str(voxels[1]) + '_z' + str(voxels[2]), header_file_fpath, np.asarray(sl_res))


def run_bootstrap_sfix():

    # change mask_fpath to point to intersection mask ###
    mask_fpath = 'int_ds_nonnanmask_0408.nii'
    mask = get_maskdata(mask_fpath)
    subj_dirpath = '../../../data/gbh/data/'
    header_file_fpath = '../../../data/gbh/data/0408161_predtrw02/filtFuncMNI_Intact_Rep1.nii'

    # mask_fpath = 'int_ds_nonnanmask_0408.nii'
    # mask = get_maskdata(mask_fpath)
    # subj_dirpath = 'all_subjs/'
    # header_file_fpath = 'all_subjs/0408161_predtrw02/filtFuncMNI_SFix_Rep1.nii'

    n_resamps = 10
    n_bstraps = 5

    ## first block gets bootstraps for scrambled fix 1, 'SFix1' ##

    # for bs in range(n_bstraps):
    #     rep1, last, diff, subjs = bootstrap(n_resamps, 'SFix1', subj_dirpath, mask, '*pred*01', randomize=True)
    #
    #     arr_to_nii('sfix1_auc_100resamps_rep1_' + str(date.today()) + '_batch' + str(bs + 1) + '.nii', header_file_fpath, rep1)
    #     arr_to_nii('sfix1_auc_100resamps_last_' + str(date.today()) + '_batch' + str(bs + 1) + '.nii', header_file_fpath, last)
    #     arr_to_nii('sfix1_auc_100resamps_diffs_' + str(date.today()) + '_batch' + str(bs + 1) + '.nii', header_file_fpath, diff)
    #
    #     assert n_resamps == len(subjs)
    #
    #     rep1_intsfix, last_intsfix, diff_intsfix, resamped_subjs = bootstrap(len(subjs), 'Intact', subj_dirpath, mask,
    #                                                                          '*pred*01', subjs=subjs)
    #
    #     arr_to_nii('intsfix1_auc_100resamps_rep1_' + str(date.today()) + '_batch' + str(bs + 1) + '.nii', header_file_fpath, rep1_intsfix)
    #     arr_to_nii('intsfix1_auc_100resamps_last_' + str(date.today()) + '_batch' + str(bs + 1) + '.nii', header_file_fpath, last_intsfix)
    #     arr_to_nii('intsfix1_auc_100resamps_diffs_' + str(date.today()) + '_batch' + str(bs + 1) + '.nii', header_file_fpath, diff_intsfix)
    #
    # print('done bootstrapping sfix1')




    ## second block gets bootstraps for scrambled fixed 2, 'SFix2' ##


    for bs in range(n_bstraps):


        rep1, last, diff, subjs = bootstrap(n_resamps, 'SFix2', subj_dirpath, mask, '*pred*02', randomize=True)

        arr_to_nii('sfix2_auc_100resamps_rep1_' + str(date.today()) + '_batch' + str(bs + 6) + '.nii', header_file_fpath, rep1)
        arr_to_nii('sfix2_auc_100resamps_last_' + str(date.today()) + '_batch' + str(bs + 6) + '.nii', header_file_fpath, last)
        arr_to_nii('sfix2_auc_100resamps_diffs_' + str(date.today()) + '_batch' + str(bs + 6) + '.nii', header_file_fpath, diff)

        assert n_resamps == len(subjs)

        rep1_intsfix, last_intsfix, diff_intsfix, resamped_subjs = bootstrap(len(subjs), 'Intact', subj_dirpath, mask, '*pred*02', subjs=subjs)

        arr_to_nii('intsfix2_auc_100resamps_rep1_' + str(date.today()) + '_batch' + str(bs + 1) + '.nii', header_file_fpath, rep1_intsfix)
        arr_to_nii('intsfix2_auc_100resamps_last_' + str(date.today()) + '_batch' + str(bs + 1) + '.nii', header_file_fpath, last_intsfix)
        arr_to_nii('intsfix2_auc_100resamps_diffs_' + str(date.today()) + '_batch' + str(bs + 1) + '.nii', header_file_fpath, diff_intsfix)

    print('done bootstrapping sfix2')


def sfix_original():


    subj_dirpath = '../../../data/gbh/data/'
    header_file_fpath = '../../../data/gbh/data/0408161_predtrw02/filtFuncMNI_Intact_Rep1.nii'

    mask_fpath = 'int_ds_nonnanmask_0408.nii'
    mask = get_maskdata(mask_fpath)


    ds_sfix1 = Dataset(subj_dirpath)
    ds_sfix1.data = get_repdata(ds_sfix1.fpath, '*pred*01', 'filt*SFix*')
    arr_to_nii('sfix1_zs.nii', header_file_fpath, np.asarray(ds_sfix1.data))

    ds_sfix2 = Dataset(subj_dirpath)
    ds_sfix2.data = get_repdata(ds_sfix2.fpath, '*pred*02', 'filt*SFix*')
    arr_to_nii('sfix2_zs.nii', header_file_fpath, np.asarray(ds_sfix2.data))

    # get searchlight results, AUCs, and 3d voxel maps of SFix1 data#

    sl_res_sfix1, sl_vox_sfix1 = s_light(ds_sfix1.data, mask, 'SFix')
    sl_sfix1_aucs = [get_AUCs(7, 2, segs) for segs in sl_res_sfix1]

    vox3d_rep1_sfix1, vox3d_lastreps_sfix1 = get_vox_map(sl_sfix1_aucs, sl_vox_sfix1, mask)
    vox_auc_diffs_sfix1 = vox3d_lastreps_sfix1 - vox3d_rep1_sfix1

    arr_to_nii('vox3d_rep1AUC_sfix1_slight_' + str(date.today()) + '.nii', header_file_fpath, vox3d_rep1_sfix1.T)
    arr_to_nii('vox3d_lastrepsAUC_sfix1_slight_' + str(date.today()) + '.nii', header_file_fpath, vox3d_lastreps_sfix1.T)
    arr_to_nii('voxAUCdiffs_sfix1_slight_' + str(date.today()) + '.nii', header_file_fpath, vox_auc_diffs_sfix1.T)

    # get searchlight results, AUCs, and 3d voxel maps of SFix2 data#

    sl_res_sfix2, sl_vox_sfix2 = s_light(ds_sfix2.data, mask, 'SFix')
    sl_sfix2_aucs = [get_AUCs(7, 2, segs) for segs in sl_res_sfix2]

    vox3d_rep1_sfix2, vox3d_lastreps_sfix2 = get_vox_map(sl_sfix2_aucs, sl_vox_sfix2, mask)
    vox_auc_diffs_sfix2 = vox3d_lastreps_sfix2 - vox3d_rep1_sfix2

    arr_to_nii('vox3d_rep1AUC_sfix2_slight_' + str(date.today()) + '.nii', header_file_fpath, vox3d_rep1_sfix2.T)
    arr_to_nii('vox3d_lastrepsAUC_sfix2_slight_' + str(date.today()) + '.nii', header_file_fpath,
               vox3d_lastreps_sfix2.T)
    arr_to_nii('voxAUCdiffs_sfix2_slight_' + str(date.today()) + '.nii', header_file_fpath, vox_auc_diffs_sfix2.T)




    # now for the split intact datasets of sfix1 and 2#

    ds_intsfix1 = Dataset(subj_dirpath)
    ds_intsfix1.data = get_repdata(ds_intsfix1.fpath, '*pred*01', 'filt*Intact*')

    ds_intsfix2 = Dataset(subj_dirpath)
    ds_intsfix2.data = get_repdata(ds_intsfix2.fpath, '*pred*02', 'filt*Intact*')

    sl_res_intsfix1, sl_vox_intsfix1 = s_light(ds_intsfix1.data, mask, 'Intact')
    sl_intsfix1_aucs = [get_AUCs(7, 2, segs) for segs in sl_res_intsfix1]

    vox3d_rep1_intsfix1, vox3d_lastreps_intsfix1 = get_vox_map(sl_intsfix1_aucs, sl_vox_intsfix1, mask)
    vox_auc_diffs_intsfix1 = vox3d_lastreps_intsfix1 - vox3d_rep1_intsfix1

    arr_to_nii('vox3d_rep1AUC_intsfix1_slight_' + str(date.today()) + '.nii', header_file_fpath, vox3d_rep1_intsfix1.T)
    arr_to_nii('vox3d_lastrepsAUC_intsfix1_slight_' + str(date.today()) + '.nii', header_file_fpath,
               vox3d_lastreps_intsfix1.T)
    arr_to_nii('voxAUCdiffs_intsfix1_slight_' + str(date.today()) + '.nii', header_file_fpath, vox_auc_diffs_intsfix1.T)

    sl_res_intsfix2, sl_vox_intsfix2 = s_light(ds_intsfix2.data, mask, 'Intact')
    sl_intsfix2_aucs = [get_AUCs(7, 2, segs) for segs in sl_res_intsfix2]

    vox3d_rep1_intsfix2, vox3d_lastreps_intsfix2 = get_vox_map(sl_intsfix2_aucs, sl_vox_intsfix2, mask)
    vox_auc_diffs_intsfix2 = vox3d_lastreps_intsfix2 - vox3d_rep1_intsfix2

    arr_to_nii('vox3d_rep1AUC_intsfix2_slight_' + str(date.today()) + '.nii', header_file_fpath, vox3d_rep1_intsfix2.T)
    arr_to_nii('vox3d_lastrepsAUC_intsfix2_slight_' + str(date.today()) + '.nii', header_file_fpath,
               vox3d_lastreps_intsfix2.T)
    arr_to_nii('voxAUCdiffs_intsfix2_slight_' + str(date.today()) + '.nii', header_file_fpath, vox_auc_diffs_intsfix2.T)




def int_original():


    # mask_fpath = 'MNI152_T1_brain_resample.nii'
    # subj_dirpath = 'all_subjs/'
    # header_file_fpath = 'all_subjs/0408161_predtrw02/filtFuncMNI_Intact_Rep1.nii'

    # ***************************************************** #

    # ***************************************************** #

    mask_fpath = '../../../data/gbh/data/MNI152_T1_brain_resample.nii'
    subj_dirpath = '../../../data/gbh/data/'
    header_file_fpath = '../../../data/gbh/data/0408161_predtrw02/filtFuncMNI_Intact_Rep1.nii'

    # ***************************************************** #

    # ***************************************************** #
    msk_start = time.time()
    mask = Dataset(mask_fpath)
    mask.data = get_maskdata(mask.fpath)
    msk_end = time.time()
    print("minutes to process mask = ", round((msk_end - msk_start) / 60, 3))

    dset_start = time.time()
    ds = Dataset(subj_dirpath)
    ds.data = get_repdata(ds.fpath, '*pred*', 'filt*Intact*')


    ds.non_nan_mask = get_non_nan_mask(ds.num_not_nan, mask.data)
    arr_to_nii('int_ds_nonnanmask_0408.nii', header_file_fpath, np.asarray(ds.non_nan_mask).T)

    dset_end = time.time()
    print("minutes to process dataset = ", round((dset_end - dset_start) / 60, 3))

    arr_to_nii('int_ds_zs_' + str(date.today()) + '.nii', header_file_fpath, np.asarray(ds.data))
    arr_to_nii('int_ds_notnan_rep1' + str(date.today()) + '.nii', header_file_fpath, np.asarray(ds.num_not_nan[0]).T)

    sl_res, sl_vox = s_light(ds.data, ds.non_nan_mask, 'Intact')

    print("number of searchlights = ", len(sl_res))

    sl_aucs = [get_AUCs(7, 2, segs) for segs in sl_res]
    vox3d_rep1, vox3d_lastreps = get_vox_map(sl_aucs, sl_vox, ds.non_nan_mask)

    arr_to_nii('vox3d_rep1AUC_int_slight_' + str(date.today()) + '.nii', header_file_fpath, vox3d_rep1.T)
    arr_to_nii('vox3d_lastrepsAUC_int_slight_' + str(date.today()) + '.nii', header_file_fpath, vox3d_lastreps.T)

    vox_AUCdiffs = vox3d_lastreps - vox3d_rep1
    arr_to_nii('voxAUCdiffs_int_slight_' + str(date.today()) + '.nii', header_file_fpath, vox_AUCdiffs.T)


def test_bootstrap():

    mask_fpath = 'MNI152_T1_brain_resample.nii'
    mask = get_maskdata(mask_fpath)
    subj_dirpath = 'all_subjs/'
    header_file_fpath = 'all_subjs/0408161_predtrw02/filtFuncMNI_Intact_Rep1.nii'

    ds = Dataset(subj_dirpath)
    ds.data, ds.num_not_nan = get_repdata(ds.fpath, '*pred*', 'filt*Intact*')
    ds.non_nan_mask = get_non_nan_mask(ds.num_not_nan, mask, min_subjs=1)

    sl_res, sl_vox = s_light(ds.data, ds.non_nan_mask, 'Intact')
    sl_aucs = [get_AUCs(7, 2, segs) for segs in sl_res]
    vox3d_rep1, vox3d_lastreps = get_vox_map(sl_aucs, sl_vox, ds.non_nan_mask)
    vox_AUCdiffs = vox3d_lastreps - vox3d_rep1

    subjects = [subjdir for subjdir in os.listdir(subj_dirpath) if fnmatch.fnmatch(subjdir, '*pred*')]

    rep1, last, diff = bootstrap(2, 'Intact', subj_dirpath, mask, '*pred*', header_file_fpath)

    arr_to_nii('auc_100resamps_rep1_' + str(date.today()) + '.nii', header_file_fpath, rep1)
    arr_to_nii('auc_100resamps_last_' + str(date.today()) + '.nii', header_file_fpath, last)
    arr_to_nii('auc_100resamps_diffs_' + str(date.today()) + '.nii', header_file_fpath, diff)


# def delta_int_sfix(int_regex, sfix_regex, n_files, res_dir='results/', shape=(121, 145, 121)):
#
#     int_sfix = []
#
#     for n_file in range(1, n_files + 1):
#
#         int_file = [file for file in os.listdir(res_dir) if fnmatch.fnmatch(file, int_regex + str(n_file) + '.nii')]
#         sfix_file = [file for file in os.listdir(res_dir) if fnmatch.fnmatch(file, sfix_regex + str(n_file) + '.nii')]

        # assert len(int_file) == 1
        # assert len(int_file) == len(sfix_file)
#
#         int_sfix.append(np.asarray(nib.load(res_dir + int_file[0]).get_fdata()) - np.asarray(nib.load(res_dir + sfix_file[0]).get_fdata()))
#
#     return np.concatenate(int_sfix, axis=0)

def lag_corr_analysis(vox_map_shape=(121, 145, 121), max_lags=20):

    ## generate lag correlations from all surviving searchlights ##

    from scipy.ndimage.measurements import label

    rois = nib.load('results/intact_analysis/auc_diffs_maskedby_sffdr.nii').get_fdata()
    rois[np.isnan(rois)] = 0
    roi_clusters = label(rois, structure=np.ones((3, 3, 3)))

    d_in_ind_sl = nib.load('results/ind_slight_res/ind_slres_x30_y38_z52.nii').get_fdata()[0]

    ## get this mask and compare to fdr corrected mask ##
    # header_file_fpath = 'all_subjs/0408161_predtrw02/filtFuncMNI_Intact_Rep2.nii'
    # arr_to_nii('cluster_labels.nii', header_file_fpath, roi_clusters[0])

    d_in = nib.load('int_ds_zs_0325.nii').get_fdata()

    aucs_first = np.full(shape=vox_map_shape, fill_value=np.nan)
    aucs_lasts = np.full(shape=vox_map_shape, fill_value=np.nan)

    max_lag_first = np.full(shape=vox_map_shape, fill_value=np.nan)
    max_lag_lasts = np.full(shape=vox_map_shape, fill_value=np.nan)

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


    for cluster in range(1, roi_clusters[1] + 1):

        mask = roi_clusters[0] == cluster
        cluster_data = d_in[:, :, mask]

        segs = tj_fit('Intact', cluster_data)
        aucs = get_AUCs(7, 2, segs)

        aucs_first[mask] = aucs[0]
        aucs_lasts[mask] = aucs[1]

        dts_first = get_DTs(segs[0])
        dts_lasts = get_DTs(segs[1])

        #   print/get array for cluster 12, 19,    #

        max_lag_first[mask] = np.argmax(lag_correlation(x=dts_first, y=ev_annot_convolved[1:], max_lags=max_lags)) - max_lags
        max_lag_lasts[mask] = np.argmax(lag_correlation(x=dts_lasts, y=ev_annot_convolved[1:], max_lags=max_lags)) - max_lags

        if cluster == 12:
            print('in cluster 12')
            print('lags of first = {0}'.format(lag_correlation(x=dts_first, y=ev_annot_convolved[1:], max_lags=max_lags)))
            print('lags of last = {0}'.format(lag_correlation(x=dts_lasts, y=ev_annot_convolved[1:], max_lags=max_lags)))
        if cluster == 19:
            print('in cluster 19')
            print('lags of first = {0}'.format(lag_correlation(x=dts_first, y=ev_annot_convolved[1:], max_lags=max_lags)))
            print('lags of last = {0}'.format(lag_correlation(x=dts_lasts, y=ev_annot_convolved[1:], max_lags=max_lags)))



    # header_file_fpath = 'all_subjs/0411161_predtrw02/filtFuncMNI_Intact_Rep5.nii'
    #
    # arr_to_nii('max_lag_TR_firsts.nii', header_file_fpath, max_lag_first)
    # arr_to_nii('max_lag_TR_lasts.nii', header_file_fpath, max_lag_lasts)
    # arr_to_nii('max_lag_TRs_first-lasts.nii', header_file_fpath, max_lag_first - max_lag_lasts)

def lag0_corr(vox_map_shape=(121, 145, 121)):

    from scipy.ndimage.measurements import label

    # generate clusters from rois that survived main analysis #
    rois = nib.load('results/intact_analysis/auc_diffs_maskedby_sffdr.nii').get_fdata()

    roi_clusters = deepcopy(rois)
    roi_clusters[np.isnan(roi_clusters)] = 0
    roi_clusters = label(roi_clusters, structure=np.ones((3, 3, 3)))

    roi_dilated = binary_dilation(roi_clusters[0], structure=np.ones((5, 5, 5), dtype=bool)).astype(int)
    roi_dilated = label(roi_dilated, structure=np.ones((3, 3, 3)))
    #
    header_file_fpath = 'all_subjs/0408161_predtrw02/filtFuncMNI_Intact_Rep2.nii'
    arr_to_nii('dilated_clusters.nii', header_file_fpath, roi_dilated[0])

    # import sys
    # sys.exit(4)

    d_in = nib.load('int_ds_zs_0325.nii').get_fdata()

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

    dil_first_lag0corr = np.full(shape=vox_map_shape, fill_value=np.nan)
    dil_lasts_lag0corr = np.full(shape=vox_map_shape, fill_value=np.nan)

    first_lag0corr = np.full(shape=vox_map_shape, fill_value=np.nan)
    lasts_lag0corr = np.full(shape=vox_map_shape, fill_value=np.nan)

    for cluster in range(1, roi_dilated[1] + 1):

        mask = np.logical_and(roi_dilated[0] == cluster, roi_clusters[0])
        cluster_data = d_in[:, :, mask]

        segs = tj_fit('Intact', cluster_data)

        dts_first = get_DTs(segs[0])
        dts_lasts = get_DTs(segs[1])

        first_lag0corr[mask] = pearsonr(dts_first, ev_annot_convolved[1:])[0]
        lasts_lag0corr[mask] = pearsonr(dts_lasts, ev_annot_convolved[1:])[0]

        dil_mask = roi_dilated[0] == cluster
        dil_clust_data = d_in[:, :, dil_mask]

        dil_segs = tj_fit(label, dil_clust_data)

        dil_dts_first = get_DTs(dil_segs[0])
        dil_dts_lasts = get_DTs(dil_segs[1])

        dil_first_lag0corr[dil_mask] = pearsonr(dil_dts_first, ev_annot_convolved[1:])[0]
        dil_lasts_lag0corr[dil_mask] = pearsonr(dil_dts_lasts, ev_annot_convolved[1:])[0]

        #
        # if cluster == 22:
        #
        #     dil_segs_arr = np.asarray(dil_segs)
        #
        #     arr_to_nii('ind_slres_cluster22_', header_file_fpath, np.asarray(dil_segs_arr))
        #     import sys
        #     sys.exit(5)



    #
    # arr_to_nii('lag0corrs_first_original.nii', header_file_fpath, first_lag0corr)
    # arr_to_nii('lag0corrs_lasts_original.nii', header_file_fpath, lasts_lag0corr)
    # arr_to_nii('dilroi_lag0corrs_first_original.nii', header_file_fpath, dil_first_lag0corr)
    # arr_to_nii('dilroi_lag0corrs_lasts_original.nii', header_file_fpath, dil_lasts_lag0corr)
    # arr_to_nii('roi_clusters_mask.nii', header_file_fpath, roi_clusters[0])

def lag0_bootstrap(n_iter):

    # before running on server: change filepaths, change # of iterations, AND CHANGE # of SUBJECTS IN BOOTSTRAP #

    # subj_dirpath = 'all_subjs/'
    header_file_fpath = 'all_subjs/0408161_predtrw02/filtFuncMNI_Intact_Rep1.nii'

    dil_cluster_mask = 'dilated_clusters.nii'
    cluster_mask = 'roi_clusters_mask.nii'

    # subj_dirpath = '../../../data/gbh/data/'
    # header_file_fpath = '../../../data/gbh/data/0408161_predtrw02/filtFuncMNI_Intact_Rep1.nii'

    for iter in range(n_iter):

        # n_resamp, condition, dil_cluster_mask_fpath, cluster_mask_fpath, n_subjs=30

        first_lag0corr, lasts_lag0corr, dil_first_lag0corr, dil_lasts_lag0corr = bootstrap_lagcorrs(10, 'Intact', dil_cluster_mask, cluster_mask)

        arr_to_nii('firstlag0_batch' + str(iter + 1) + '_' + str(date.today()) + '.nii', header_file_fpath, first_lag0corr)
        arr_to_nii('lastslag0_batch' + str(iter + 1) + '_' + str(date.today()) + '.nii', header_file_fpath, lasts_lag0corr)
        arr_to_nii('dil_firstlag0_batch' + str(iter + 1) + '_' + str(date.today()) + '.nii', header_file_fpath, dil_first_lag0corr)
        arr_to_nii('dil_lastslag0_batch' + str(iter + 1) + '_' + str(date.today()) + '.nii', header_file_fpath, dil_lasts_lag0corr)

def save_bstrap_clusters(res_dir, bstraps_regex, x, y, z, bstrap_type):

    bstrap_files = [file for file in os.listdir(res_dir) if fnmatch.fnmatch(file, bstraps_regex)]
    bstraps = np.concatenate([np.asarray(nib.load(res_dir + file).get_fdata()) for file in bstrap_files], axis=0)

    header_file_fpath = 'all_subjs/0408161_predtrw02/filtFuncMNI_Intact_Rep1.nii'

    arr_to_nii('bstrap_res_x=' + str(x) + 'y=' + str(y) + 'z=' + str(z) + bstrap_type, header_file_fpath, np.asarray(bstraps[:, x, y, z]))