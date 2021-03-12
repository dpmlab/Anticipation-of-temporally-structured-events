import glob
import nibabel as nib
import numpy as np
import pickle
from s_light import load_pickle

data_fpath = '../data/'
output_fpath = '../outputs/perm/'
subjects = glob.glob(data_fpath + '*pred*')
header_fpath = data_fpath + 'header.nii'


non_nan = nib.load(data_fpath + 'valid_vox.nii').get_fdata().T > 0
with open (data_fpath + 'SL/SL_allvox.p', 'rb') as fp:
    SL_allvox = pickle.load(fp)

#load_pickle(0, output_fpath + '/pickles/', non_nan, SL_allvox, header_fpath, '../outputs/AUC.nii')
#load_pickle(1, output_fpath + '/pickles/', non_nan, SL_allvox, header_fpath, '../outputs/AUC_S10.nii')
#load_pickle(2, output_fpath + '/pickles/', non_nan, SL_allvox, header_fpath, '../outputs/AUC_jointfit')
load_pickle(3, output_fpath + '/pickles/', non_nan, SL_allvox, header_fpath, '../outputs/AUC_tuneK.nii')
#load_pickle(4, output_fpath + '/pickles/', non_nan, SL_allvox, header_fpath, '../outputs/AUC_SF')
