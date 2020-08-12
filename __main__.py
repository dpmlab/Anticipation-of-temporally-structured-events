import sys, time, os, fnmatch
from datetime import date

from experiments import conf_int_bootstrap_res, run_bootstrap, pdf_sf_bootstrap_res, ks_norm, get_fdr, get_individual_sl_res, run_bootstrap_sfix, get_fdr_sf_maskedvals, lag_corr_analysis, lag0_corr, lag0_bootstrap
from utils.pickles import arr_to_nii, split_nii, diff_int_sfix, means_bstraps

import nibabel as nib, numpy as np

from analysis import lag_correlation

from utils.Dataset import get_rois




def main():

    # print("mem snapshot after call to main")
    # get_mem_snapshot()

    # 12/13/18 meeting: 50 - 100 bootstraps, save data with header info, get count of how many subjects data per SL/vox #

    # get command line flags (see JupNB for specs) #
    # easier to add order in README or just declare new vars? #

    header_file_fpath = 'all_subjs/0408161_predtrw02/filtFuncMNI_Intact_Rep2.nii'
    today = str(date.today())

    # sfix1_orig = 'results/voxAUCdiffs_sfix1_slight_2019-06-07.nii'
    # sfix2_orig = 'results/voxAUCdiffs_sfix2_slight_2019-06-08.nii'
    # avg_sfix12 = np.asarray(nib.load(sfix1_orig).get_fdata())
    # avg_sfix12 += np.asarray(nib.load(sfix2_orig).get_fdata())
    # avg_sfix12 /= 2
    # arr_to_nii('avg_sfix12_orig_bstraps.nii', header_file_fpath, avg_sfix12)
    # avg_fpath = 'avg_sfix12_orig_bstraps.nii'
    #
    # sfix_bs_reg = 'avg_sfix12*.nii'
    # avg_fpath = 'int1minussfix1_originalaucs.nii'
    # sfix_bs_reg = 'int1*resamp*.nii'
    # avg_fpath = 'int2minussfix2_originalaucs.nii'
    # sfix_bs_reg = 'int2*resamp*.nii'
    #

    avg_fpath = 'avg_intsfix12_original.nii'
    sfix_bs_reg = 'avg*.nii'
    conf_int_res, in_conf_int, conf_int_min, conf_int_max, n_less0, conf_min_val, conf_mean, conf_std, conf_var, conf_pval = conf_int_bootstrap_res(avg_fpath, sfix_bs_reg, res_dir='results/intsfix_avg_analysis/')

    # arr_to_nii('unfiltered_avg_intsfix12_bstraps_less0_' + today + '_100bs.nii', header_file_fpath, n_less0.T)
    # arr_to_nii('unfiltered_avg_intsfix12_min_val_conf_int_' + today + '_100bs.nii', header_file_fpath, conf_min_val.T)
    # arr_to_nii('unfiltered_avg_intsfix12_mean_95confint_' + today + '_100bs.nii', header_file_fpath, conf_mean.T)
    # arr_to_nii('unfiltered_avg_intsfix12_std_95confint_' + today + '_100bs.nii', header_file_fpath, conf_std.T)
    # arr_to_nii('unfiltered_avg_intsfix12_var_95confint_' + today + '_100bs.nii', header_file_fpath, conf_var.T)
    # arr_to_nii('unfiltered_avg_intsfix12_p95_95confint_' + today + '_100bs.nii', header_file_fpath, conf_pval.T)

    pdf_res, sf_res, cv_sf_res = pdf_sf_bootstrap_res(avg_fpath, sfix_bs_reg, res_dir='results/intsfix_avg_analysis/')
    #
    # arr_to_nii('unfiltered_avg_intsfix12_pdf_' + today + '_100bs.nii', header_file_fpath, pdf_res.T)
    # arr_to_nii('unfiltered_avg_intsfix12_sf_' + today + '_100bs.nii', header_file_fpath, sf_res.T)
    # arr_to_nii('unfiltered_avg_intsfix12_cv_sf_' + today + '_100bs.nii', header_file_fpath, cv_sf_res.T)

    sys.exit(5)


    #if reading in files, need to transpose again....#

    get_fdr_sf_maskedvals(avg_fpath, 'unfiltered_avg_intsfix12_cv_sf_' + today + '_100bs.nii', 'unfiltered_avg_intsfix12_p95_95confint_' + today + '_100bs.nii', 'unfiltered_avg_intsfix12_sf_fdr_' + today + '.nii', 'unfiltered_avg_intsfix12_p95_fdr_' + today + '.nii', 'unfiltered_avg_intsfix12_aucs_masked_sffdr_' + today + '.nii', 'unfiltered_avg_intsfix12_aucs_masked_p95fdr_' + today + '.nii', header_file_fpath)

    # dset_int = npz_to_array('results/pickles/dset.npz')
    # dset_int = dset_int[0]

    # arr_to_hdf5('dset_int_ds', dset_int.data)
    # arr_to_hdf5('dset_int_nan', dset_int.num_not_nan)

    #dset_int = hdf5_to_arr('dset_int_ds.h5')


    # print("mem snapshot after reading in mask and dataset and saving nifties of z-scored datasets")
    # get_mem_snapshot()
    #
    #
    #
    # ds = Dataset('orig_ds_zs.nii')
    # ds.data = nib.load('orig_ds_zs.nii').get_fdata()
    # ds.num_not_nan = nib.load('org_ds_notnan_rep1_0301.nii').get_fdata()
    # ds.non_nan_mask = get_non_nan_mask(ds.num_not_nan, mask.data)
    #
    # arr_to_nii('intact_non_nan_mask.nii', header_file_fpath, ds.non_nan_mask)


    # print("mem snapshot after searchlight on original ds and saving to nifties")
    # get_mem_snapshot()

    # malloc pointers #
    # sl_res = None
    # sl_vox = None
    # dset_int = None

    #bootstrap(50, 'intact', subj_dirpath, mask, '*pred*', header_file_fpath)


def get_mem_snapshot():
    snapshot = tracemalloc.take_snapshot()
    stats = snapshot.statistics('lineno')
    print("**********************************************************************")
    for stat in stats[:10]:
        print(stat)
    print("**********************************************************************")






if __name__ == '__main__':

    # import tracemalloc
    # tracemalloc.start()
    #
    # try:
    #     main()
    # except MemoryError:
    #     print("mem snapshot after catching MemoryError")
    #     get_mem_snapshot()

    # ev_annot = np.asarray(
    #     [5, 12, 54, 77, 90,
    #      3, 12, 23, 30, 36, 43, 50, 53, 78, 81, 87, 90,
    #      11, 23, 30, 50, 74,
    #      1, 55, 75, 90,
    #      4, 10, 53, 77, 82, 90,
    #      11, 54, 77, 81, 90,
    #      12, 22, 36, 54, 78,
    #      12, 52, 79, 90,
    #      10, 23, 30, 36, 43, 50, 77, 90,
    #      13, 55, 79, 90,
    #      4, 10, 23, 29, 35, 44, 51, 56, 77, 80, 85, 90,
    #      11, 55, 78, 90,
    #      11, 30, 43, 54, 77, 90,
    #      4, 11, 24, 30, 38, 44, 54, 77, 90]
    # )

    header_file_fpath = 'all_subjs/0411161_predtrw02/filtFuncMNI_Intact_Rep5.nii'
    #
    # res_dir = 'results/'
    #
    # bstraps_regexsfix1 = 'sfix1*batch*.nii'
    # bstraps_regexsfix2 = 'sfix2*batch*.nii'
    #
    # resampsfix1_files = [file for file in os.listdir(res_dir) if fnmatch.fnmatch(file, bstraps_regexsfix1)]
    # resampsfix2_files = [file for file in os.listdir(res_dir) if fnmatch.fnmatch(file, bstraps_regexsfix2)]
    #
    # resampsfix1_aucs = np.concatenate([np.asarray(nib.load(res_dir + file).get_fdata()) for file in resampsfix1_files], axis=0)
    # resampsfix2_aucs = np.concatenate([np.asarray(nib.load(res_dir + file).get_fdata()) for file in resampsfix2_files], axis=0)
    #
    # resampsfix1_aucs += resampsfix2_aucs
    # resampsfix1_aucs /= 2
    #
    # arr_to_nii('avg_sfix12_resampaucs.nii', header_file_fpath, resampsfix1_aucs)
    #

    # temp_sfix2 = delta_int_sfix('sfix2*batch*', 'intsfix2*batch*', 10)
    # arr_to_nii('int2minussfix2_resampaucs.nii', header_file_fpath, temp_sfix2)
    # temp_sfix1 = delta_int_sfix('sfix1*batch*', 'intsfix1*batch*', 10)
    # arr_to_nii('int1minussfix1_resampaucs.nii', header_file_fpath, temp_sfix1)

    # intsfix1 = np.asarray(nib.load('results/voxAUCdiffs_intsfix1_slight_2019-06-08.nii').get_fdata())
    # intsfix2 = np.asarray(nib.load('results/voxAUCdiffs_intsfix2_slight_2019-06-08.nii').get_fdata())
    #
    # sfix1 = np.asarray(nib.load('results/voxAUCdiffs_sfix1_slight_2019-06-07.nii').get_fdata())
    # sfix2 = np.asarray(nib.load('results/voxAUCdiffs_sfix2_slight_2019-06-08.nii').get_fdata())
    #
    # arr_to_nii('int1minussfix1_originalaucs.nii', header_file_fpath, intsfix1 - sfix1)
    # arr_to_nii('int2minussfix2_originalaucs.nii', header_file_fpath, intsfix2 - sfix2)


    # int1sfix1 = np.asarray(nib.load('int1minussfix1_originalaucs.nii').get_fdata())
    # int2sfix2 = np.asarray(nib.load('int2minussfix2_originalaucs.nii').get_fdata())
    #
    # int1sfix1 += int2sfix2
    # int1sfix1 /= 2
    # arr_to_nii('avg_int12minussfix12_originalaucs.nii', header_file_fpath, int1sfix1)
    #
    # isf1_resamps = np.asarray(nib.load('results/int1minussfix1_resampaucs.nii').get_fdata())
    # isf2_resamps = np.asarray(nib.load('results/int2minussfix2_resampaucs.nii').get_fdata())
    #
    # isf1_resamps += int2sfix2
    # isf1_resamps /= 2
    #
    # arr_to_nii('avg_int12minussfix12_resampaucs.nii', header_file_fpath, isf1_resamps)
    #
    #
    #
    # split_nii('results/resamps/sfix2_auc_100resamps_diffs_2019-06-11_batch1.nii', 'sf2T_resamp_split', header_file_fpath, transpose=True)
    # sys.exit(3)

    # 07/08 debugging/trouble-shooting #

    # diff_int_sfix('intsfix1*batch*', 'sfix1*batch*', 10, 'results/resamps/', 'int1-sfix1', header_file_fpath)
    # diff_int_sfix('intsfix2*batch*', 'sfix2*batch*', 10, 'results/resamps/', 'int2-sfix2', header_file_fpath)

    # split_nii('int1-sfix1_1.nii', 'splitint1sf1', header_file_fpath, transpose=True)
    # sys.exit(4)
    #

    # res_dir = 'results/intminussfix/resamps/'

    # bstraps_regex = 'int1*sfix1*.nii'
    #
    # resamp_files = [file for file in os.listdir(res_dir) if fnmatch.fnmatch(file, bstraps_regex)]
    # resamp_aucs = np.concatenate([np.asarray(nib.load(res_dir + file).get_fdata()) for file in resamp_files], axis=0)
    #
    # means = np.nanmean(resamp_aucs, axis=0)
    # meds = np.nanmedian(resamp_aucs, axis=0)
    #
    # arr_to_nii('mean_int1-sfix1.nii', header_file_fpath, means.T)
    # arr_to_nii('median_int1-sfix1.nii', header_file_fpath, meds.T)

    # means_bstraps(g1_regex='int1*sfix1*', g2_regex='int2*sfix2*', n_files=10, outpath='avg_int12-sfix12', res_dir='results/intminussfix/resamps/', header_fpath=header_file_fpath)
    # means_bstraps(g1_regex='sfix1*', g2_regex='sfix2*', n_files=10, outpath='avg_sfix12', res_dir='results/resamps/', header_fpath=header_file_fpath)

    # means_bstraps(g1_regex='intsfix1*', g2_regex='intsfix2*', n_files=10, outpath='avg_intsfix12',
    #               res_dir='results/0603_sfix/resamps/', header_fpath=header_file_fpath)


    # intsfix1_aucs = (nib.load('results/voxAUCdiffs_intsfix1_slight_2019-06-08.nii').get_fdata())
    # intsfix2_aucs = (nib.load('results/voxAUCdiffs_intsfix2_slight_2019-06-08.nii').get_fdata())
    # intsfix1_aucs += intsfix2_aucs
    # intsfix1_aucs /= 2
    #
    # arr_to_nii('avg_intsfix12_original.nii', header_file_fpath, intsfix1_aucs)

    # temp = np.asarray(nib.load('results/ind_slight_res/ind_slres_x30_y38_z52.nii').get_fdata())[0]
    # print(temp.shape)
    #
    # auc1 = np.dot(temp[0], np.arange(7))
    # auc2 = np.dot(temp[1], np.arange(7))
    # print(auc1.shape)

    # temp1 = np.random.randint(10, size=21)
    # temp2 = np.random.randint(10, size=20)
    # max_l = 5
    #
    # corrs = lag_correlation(x=temp1, y=temp2, max_lags=max_l)
    #
    # sys.exit(4)

    # lag_corr_analysis()


    # lag0_bootstrap(10)
    lag0_corr()

    # from experiments import save_bstrap_clusters
    #
    # save_bstrap_clusters('results/', 'dil_firstlag0*', 94, 53, 49, 'dil_firstlag0')
    # save_bstrap_clusters('results/', 'dil_lastslag0*', 94, 53, 49, 'dil_lastslag0')


    # subj_dirpath = '../../../data/gbh/data/'
    # header_file_fpath = '../../../data/gbh/data/0408161_predtrw02/filtFuncMNI_Intact_Rep1.nii'
    # subj_dirpath = 'all_subjs/'
    #
    # get_rois(fpath=subj_dirpath, condition_regex='filt*Intact*', roi_mask_fpath='dilated_clusters.nii', subj_regex='*pred*')
    # from utils.Dataset import get_roidata
    #
    # d = get_roidata(roi=4, subj_idxs=[0, 0, 1, 0])

    # from utils.Dataset import get_repdata
    #
    # subj_dirpath = 'all_subjs/'
    # d_in = get_repdata(fpath=subj_dirpath, condition_regex='dil_roi*', subj_regex='*pred*', file_ext='.h5', roi_data=True)
    #
    # print('done')
    # main()

    # from experiments import get_conf_int_results
    # get_conf_int_results()

    # from experiments import lag0_diffs_bstraps
    # #
    # lag0_diffs_bstraps('firstlag0*', 'lastslag0*', 'diffs_lag0_percentpos.nii', 'diffs_lag0_percentneg.nii')
    # lag0_diffs_bstraps('dil_firstlag0*', 'dil_lastslag0*', 'diffs_dil_lag0_percentpos.nii', 'diffs_dil_lag0_percentneg.nii')

    # split_nii('results/dil_lastslag0_batch1_2019-10-10.nii', 'dillasts_bs', header_file_fpath)



