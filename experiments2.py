from datetime import date
import numpy as np

import nibabel as nib
from utils.pickles import arr_to_nii
from utils.Dataset import Dataset, get_repdata, get_maskdata


from analysis import get_AUCs

from s_light import s_light, get_vox_map


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