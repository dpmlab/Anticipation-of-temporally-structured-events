import glob
import sys
import nibabel as nib
from s_light import s_light

data_fpath = '../data/'
output_fpath = '../outputs/'
subjects = glob.glob(data_fpath + '*pred*')
header_fpath = data_fpath + 'header.nii'


non_nan = nib.load(data_fpath + 'valid_vox.nii').get_fdata().T > 0

analysis_type = int(sys.argv[1])

if analysis_type == 0:
	# Traditional
	s_light(True, False, 0, False,
	           '../data/SL/', subjects, non_nan, header_fpath, output_fpath + 'AUC.nii')
elif analysis_type == 1:
	# SRM15
	s_light(True, False, 15, False,
	           '../data/SL/', subjects, non_nan, header_fpath, output_fpath + 'AUC_S15.nii')
elif analysis_type == 2:
	# SRM10
	s_light(True, False, 10, False,
	           '../data/SL/', subjects, non_nan, header_fpath, output_fpath + 'AUC_S10.nii')
elif analysis_type == 3:
	# Joint fit 6
	s_light(False, False, 0, False,
	           '../data/SL/', subjects, non_nan, header_fpath, output_fpath + 'AUC_jointfit')
elif analysis_type == 4:
	# Tune K
	s_light(True, True, 0, False,
	          '../data/SL/', subjects, non_nan, header_fpath, output_fpath + 'AUC_tuneK.nii')
elif analysis_type == 5:
	# SFix
	s_light(True, False, 0, True,
	           '../data/SL/', subjects, non_nan, header_fpath, output_fpath + 'AUCSF')
