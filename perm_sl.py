import glob
import sys
import nibabel as nib
import numpy as np
from numpy.random import default_rng
from s_light import s_light

data_fpath = '../data/'
output_fpath = '../outputs/perm/'
subjects = glob.glob(data_fpath + '*pred*')
header_fpath = data_fpath + 'header.nii'


non_nan = nib.load(data_fpath + 'valid_vox.nii').get_fdata().T > 0

analysis_type = int(sys.argv[1])
perm_start = int(sys.argv[2])
perm_end = int(sys.argv[3])

rng = default_rng(perm_start) # Seed with perm start

for i in range(perm_start, perm_end):

	subj_perms = dict()
	for s in subjects:
		if i == 0:
			# First perm is the real analysis
			subj_perms[s] = np.arange(6)
		else:
			subj_perms[s] = rng.permutation(6)
	print(subj_perms)
	print('Running perm ', i)

	if analysis_type == 0:
		# Traditional
		s_light(True, False, 0, False,
		           '../data/SL/', subj_perms, non_nan, header_fpath, output_fpath + 'p' + str(i) + '_AUC.nii')
	elif analysis_type == 1:
		# SRM10
		s_light(True, False, 10, False,
		           '../data/SL/', subj_perms, non_nan, header_fpath, output_fpath + 'p' + str(i) + '_AUC_S10.nii')
	elif analysis_type == 2:
		# Joint fit 6
		s_light(False, False, 0, False,
		           '../data/SL/', subj_perms, non_nan, header_fpath, output_fpath + 'p' + str(i) + '_AUC_jointfit')
	elif analysis_type == 3:
		# Tune K
		s_light(True, True, 0, False,
		          '../data/SL/', subj_perms, non_nan, header_fpath, output_fpath + 'p' + str(i) + '_AUC_tuneK.nii')
	elif analysis_type == 4:
		# SFix
		s_light(True, False, 0, True,
		           '../data/SL/', subj_perms, non_nan, header_fpath, output_fpath + 'p' + str(i) + '_AUCSF')
