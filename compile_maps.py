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

# load_pickle(0, output_fpath + '/pickles/', non_nan, SL_allvox, header_fpath, '../outputs/AUC.nii')
# load_pickle(1, output_fpath + '/pickles/', non_nan, SL_allvox, header_fpath, '../outputs/AUC_S10.nii')
# load_pickle(2, output_fpath + '/pickles/', non_nan, SL_allvox, header_fpath, '../outputs/AUC_jointfit')
load_pickle(3, output_fpath + '/pickles/', non_nan, SL_allvox, header_fpath, '../outputs/AUC_tuneK')
# load_pickle(4, output_fpath + '/pickles/', non_nan, SL_allvox, header_fpath, '../outputs/AUC_SF')
load_pickle(5, output_fpath + '/pickles/', non_nan, SL_allvox, header_fpath, '../outputs/AUC_S10_jointfit')
load_pickle(6, output_fpath + '/pickles/', non_nan, SL_allvox, header_fpath, '../outputs/AUC_SF_S10_jointfit')
load_pickle(7, output_fpath + '/pickles/', non_nan, SL_allvox, header_fpath, '../outputs/AUC_usetunedK_S10_jointfit')
load_pickle(8, output_fpath + '/pickles/', non_nan, SL_allvox, header_fpath, '../outputs/corr_shift')

# Correlate K map and AUC map
AUC   = nib.load('../outputs/AUC_S10_jointfit_mean.nii').get_fdata().T
AUC_q = nib.load('../outputs/AUC_S10_jointfit_mean_q.nii').get_fdata().T
K     = nib.load('../outputs/AUC_tuneK_K.nii').get_fdata().T

mask = (AUC_q > 0)*(AUC_q < 0.05)
print(spearmanr(AUC[mask], K[mask])[0])

import matplotlib.pyplot as plt
AUC_mask = AUC[mask]
K_round = np.around(K[mask])
for k in range(2,9):
	plt.violinplot(AUC_mask[K_round == k], positions=[9-k], showextrema=False)
plt.xticks(np.arange(1,8), [round(90/(9-x)) for x in np.arange(1,8)])
plt.xlabel('Optimal event timescale (seconds)')
plt.ylabel('Prediction (seconds)')
plt.savefig('../outputs/KvsPred.png')