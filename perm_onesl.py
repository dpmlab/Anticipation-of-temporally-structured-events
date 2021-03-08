import glob
import sys
import nibabel as nib
import numpy as np
from tqdm import tqdm
import pickle
import time
import tables
import numpy as np
from numpy.random import default_rng
from s_light import one_sl, one_sl_SF

data_fpath = '../data/'
output_fpath = '../outputs/perm/'
subjects = ['../data/0814151_predtrw01', '../data/0425161_predtrw02', '../data/0904151_predtrw01', '../data/0707151_predtrw01', '../data/0421161_predtrw02', '../data/0706151_predtrw01', '../data/0902151_predtrw01', '../data/0408161_predtrw02', '../data/0724151_predtrw01', '../data/0510161_predtrw02', '../data/0511161_predtrw02', '../data/0407161_predtrw02', '../data/0504161_predtrw02', '../data/0731151_predtrw01', '../data/0622151_predtrw01', '../data/0826151_predtrw01', '../data/0827151_predtrw01', '../data/0503161_predtrw02', '../data/0624151_predtrw01', '../data/0411161_predtrw02', '../data/0502161_predtrw02', '../data/0413161_predtrw02', '../data/0821151_predtrw01', '../data/0626151_predtrw01', '../data/0803152_predtrw01', '../data/0419161_predtrw02', '../data/0418161_predtrw02', '../data/0509162_predtrw02', '../data/0509161_predtrw02', '../data/0803151_predtrw01']
header_fpath = data_fpath + 'header.nii'

sl_i = int(sys.argv[1])
perm_start = 0
perm_end = 100

rng = default_rng(perm_start) # Seed with perm start

print('Loading data...')
load_start = time.time()
sl_h5 = tables.open_file('../data/SL/' + str(sl_i) + '.h5', mode='r')

subj_list_orig = []
for subj in subjects:
    subjname = '/subj_' + subj.split('/')[-1]
    d = sl_h5.get_node(subjname, 'Intact').read()
    subj_list_orig.append(d)

Intact_subj_list_orig = dict()
SFix_subj_list_orig = dict()
for group in ['predtrw01', 'predtrw02']:
    Intact_subj_list_orig[group] = []
    SFix_subj_list_orig[group] = []
    for subj in [s for s in subjects if group in s]:
        subjname = '/subj_' + subj.split('/')[-1]
        dI = sl_h5.get_node(subjname, 'Intact').read()
        Intact_subj_list_orig[group].append(dI)
        dS = sl_h5.get_node(subjname, 'SFix').read()
        SFix_subj_list_orig[group].append(dS)

sl_h5.close()
print('  Loaded in', time.time() - load_start, 'seconds')

for analysis_type in range(5):
    print('Analysis', analysis_type)
    if analysis_type == 4:
        sl_AUCdiffs_Intact = []
        sl_AUCdiffs_SFix = []
    else:
        sl_AUCs = []
        if analysis_type == 3:
            sl_K = []

    for i in tqdm(range(perm_start, perm_end)):
        
        if analysis_type == 4:
            group_Intact = []
            group_SFix = []
            for group in ['predtrw01', 'predtrw02']:
                Intact_subj_list = []
                SFix_subj_list = []
                for s in range(len(Intact_subj_list_orig[group])):
                    if i == 0:  # Don't permute perm=0
                        subj_perm = np.arange(6)
                    else:
                        subj_perm = rng.permutation(6)
                    Intact_subj_list.append(Intact_subj_list_orig[group][s][subj_perm])
                    SFix_subj_list.append(SFix_subj_list_orig[group][s][subj_perm])
                group_Intact.append(np.mean(Intact_subj_list, axis=0))
                group_SFix.append(np.mean(SFix_subj_list, axis=0))
        else:
            subj_list = []
            for s in range(len(subj_list_orig)):
                if i == 0:  # Don't permute perm=0
                    subj_perm = np.arange(6)
                else:
                    subj_perm = rng.permutation(6)
                subj_list.append(subj_list_orig[s][subj_perm])

        if analysis_type == 0:
            # Traditional
            sl_AUCs.append(one_sl(subj_list, subjects, True, False, 0)[0])
        elif analysis_type == 1:
            # SRM10
            sl_AUCs.append(one_sl(subj_list, subjects, True, False, 10)[0])
        elif analysis_type == 2:
            # Joint fit 6
            sl_AUCs.append(one_sl(subj_list, subjects, False, False, 0)[0])
        elif analysis_type == 3:
            # Tune K
            sl_res = one_sl(subj_list, subjects, True, True, 0)
            sl_AUCs.append(sl_res[0])
            sl_K.append(sl_res[1])
        elif analysis_type == 4:
            # SFix
            sl_res = one_sl_SF(group_Intact, group_SFix, True)
            sl_AUCdiffs_Intact.append(sl_res[0])
            sl_AUCdiffs_SFix.append(sl_res[1])

    with open(output_fpath + '/pickles/_' + str(analysis_type) + '_' + str(sl_i) + '_' + str(perm_start) + '_' + str(perm_end) +'_.p', 'wb') as fp:
        if analysis_type == 4:
            pickle.dump((sl_AUCdiffs_Intact, sl_AUCdiffs_SFix), fp)
        elif analysis_type == 3:
            pickle.dump((sl_AUCs, sl_K), fp)
        else:
            pickle.dump(sl_AUCs, fp)
