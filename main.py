import glob
from copy import deepcopy
import random
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.measurements import label
import nibabel as nib
import numpy as np
from Dataset import Dataset, save_rois
from utils import ev_annot_freq, hrf_convolution, bootstrap_stats, mask_nii
from lagcorr import lag_corr
from s_light import run_s_light_auc

data_fpath = '../data/'
output_fpath = '../outputs/'
subjects = glob.glob(data_fpath + '*pred*')
header_fpath = '../data/0411161_predtrw02/filtFuncMNI_Intact_Rep5.nii'


# Load full (non-bootstrapped) dataset
dset = Dataset()
dset.load_data(data_fpath, subjects, 'filt*Intact*')
valid_vox = deepcopy(dset.non_nan_mask)


# Run searchlight analysis
run_s_light_auc(dset, output_fpath + 'AUC.nii', header_fpath)


# Run searchlight bootstraps
n_resamp = 100
for resamp in range(n_resamp):
    resamp_subjs = [random.choice(subjects) for s in range(len(subjects))]
    dset.load_data(data_fpath, resamp_subjs, 'filt*Intact*')
    dset.non_nan_mask = valid_vox
    bootpath = output_fpath + 'boot/AUC_boot' + str(resamp) + '.nii'
    run_s_light_auc(dset, bootpath, header_fpath)
bootstrap_stats(output_fpath + 'boot/AUC_boot*.nii',
                output_fpath + 'AUC_q.nii')

# Mask analysis by bootstrapped statistics
mask_nii(output_fpath + 'AUC.nii', output_fpath + 'AUC_q.nii',
         output_fpath + 'AUC_q05.nii')


# Identify significant clusters
AUC_results = nib.load(output_fpath + 'AUC_q05.nii').get_fdata().T
clusters = label(AUC_results, structure=np.ones((3, 3, 3)))[0]
rois = binary_dilation(clusters,
                       structure=np.ones((5, 5, 5), dtype=bool)).astype(int)
rois = label(rois, structure=np.ones((3, 3, 3)))[0]


# Save ROI datasets
save_rois(data_fpath, subjects, 'filt*Intact*', rois > 0,
          '../outputs/rois/')

# Run lag correlation analysis
max_lag = 7
ev_conv = hrf_convolution(ev_annot_freq())
dset.load_rois('../outputs/rois/', subjects)
first_lagcorr, lasts_lagcorr = lag_corr(dset, rois, ev_conv, max_lag,
                                        header_fpath, output_fpath + 'lagcorr')

# Run lag correlation bootstraps
n_resamp = 100
for resamp in range(n_resamp):
    resamp_subjs = [random.choice(subjects) for s in range(len(subjects))]
    dset.load_rois('../outputs/rois/', resamp_subjs)
    bootpath = output_fpath + 'boot/lagcorr_boot' + str(resamp)
    lag_corr(dset, rois, ev_conv, max_lag, header_fpath, bootpath)
bootstrap_stats(output_fpath + 'boot/lagcorr_boot*first.nii',
                output_fpath + 'lagcorr_first_p.nii', use_z = False)
bootstrap_stats(output_fpath + 'boot/lagcorr_boot*lasts.nii',
                output_fpath + 'lagcorr_last_p.nii', use_z = False)
bootstrap_stats(output_fpath + 'boot/lagcorr_boot*diff.nii',
                output_fpath + 'lagcorr_diff_p.nii', use_z = False)
