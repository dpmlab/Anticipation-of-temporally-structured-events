import glob
import nibabel as nib
import numpy as np
import pickle
from s_light import load_pickle
from scipy.stats import spearmanr

data_fpath = '../data/'
output_fpath = '../outputs/perm/'
subjects = glob.glob(data_fpath + '*pred*')
header_fpath = data_fpath + 'header.nii'


non_nan = nib.load(data_fpath + 'valid_vox.nii').get_fdata().T > 0
with open (data_fpath + 'SL/SL_allvox.p', 'rb') as fp:
    SL_allvox = pickle.load(fp)

#load_pickle(3, output_fpath + '/pickles/', non_nan, SL_allvox, header_fpath, '../outputs/AUC_tuneK')
load_pickle(5, output_fpath + '/pickles/', non_nan, SL_allvox, header_fpath, '../outputs/AUC_S10_jointfit')
#load_pickle(6, output_fpath + '/pickles/', non_nan, SL_allvox, header_fpath, '../outputs/AUC_SF_S10_jointfit')
#load_pickle(7, output_fpath + '/pickles/', non_nan, SL_allvox, header_fpath, '../outputs/AUC_usetunedK_S10_jointfit')
#load_pickle(8, output_fpath + '/pickles/', non_nan, SL_allvox, header_fpath, '../outputs/corr_shift')
