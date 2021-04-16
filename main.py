import glob
import pickle
import tables
import nibabel as nib
import numpy as np
from numpy.random import default_rng
from data import find_valid_vox, save_s_lights
from s_light import optimal_events, compile_optimal_events, \
                    fit_HMM, compile_fit_HMM, \
                    shift_corr, compile_shift_corr

nSL = 5354
nPerm = 100
max_lag = 10
subjects = glob.glob('../data/*predtrw*')
header_fpath = '../data/0407161_predtrw02/filtFuncMNI_Intact_Rep1.nii'

# Create valid_vox.nii mask
find_valid_vox('../data/', subjects)

# Create a separate data file for each searchlight
non_nan = nib.load('../data/valid_vox.nii').get_fdata().T > 0
save_s_lights('../data/', non_nan, '../data/SL/')

# Run all analyses in each searchlight
# This will take ~1000 CPU hours, and so should be run
# in parallel on a cluster if possible
for sl_i in range(nSL):

    # Load data for this searchlight
    sl_h5 = tables.open_file('../data/SL/%d.h5' % sl_i, mode='r')
    data_list_orig = []
    for subj in subjects:
        subjname = '/subj_' + subj.split('/')[-1]
        d = sl_h5.get_node(subjname, 'Intact').read()
        data_list_orig.append(d)
    sl_h5.close()
    nSubj = len(data_list_orig)


    sl_K = []
    sl_seg = []
    sl_shift_corr = []
    rng = default_rng(0)
    # Repeat analyses for each permutation
    for p in range(nPerm):
        data_list = []
        for s in range(nSubj):
            if p == 0:
                # This is the real (non-permuted) analysis
                subj_perm = np.arange(6)
            else:
                subj_perm = rng.permutation(6)
            data_list.append(data_list_orig[s][subj_perm])

        # Run all three analysis types
        sl_K.append(optimal_events(data_list, subjects))
        sl_seg.append(fit_HMM(data_list))
        sl_shift_corr.append(shift_corr(data_list, max_lag))

    # Save results for this searchlight
    pickle.dump(sl_K,
                open('../outputs/perm/optimal_events_%d.p' % sl_i, 'wb'))
    pickle.dump(sl_seg,
                open('../outputs/perm/fit_HMM_%d.p' % sl_i, 'wb'))
    pickle.dump(sl_shift_corr,
                open('../outputs/perm/shift_corr_%d.p' % sl_i, 'wb'))

# Compile results into final maps
SL_allvox = pickle.load(open('../data/SL/SL_allvox.p', 'rb'))
compile_optimal_events('../outputs/perm/', non_nan, SL_allvox,
                        header_fpath, '../outputs/')
opt_event = nib.load('optimal_events.nii').get_fdata().T
compile_fit_HMM('../outputs/perm/', non_nan, SL_allvox,
                header_fpath, '../outputs/', opt_event)
compile_shift_corr('../outputs/perm/', non_nan, SL_allvox,
                   header_fpath, '../outputs')
