import glob
import nibabel as nib
from s_light import s_light

data_fpath = '../data/'
output_fpath = '../outputs/'
subjects = glob.glob(data_fpath + '*pred*')
header_fpath = '../data/0411161_predtrw02/filtFuncMNI_Intact_Rep5.nii'


non_nan = nib.load('../code_preprint/valid_vox.nii').get_fdata().T > 0
#s_light('../data/SL/', subjects, non_nan, header_fpath, output_fpath + 'AUC_SRM15.nii')

s_light(True, 0, '../data/SL/', subjects, non_nan, header_fpath, output_fpath + 'AUC.nii')
s_light(True, 15, '../data/SL/', subjects, non_nan, header_fpath, output_fpath + 'AUC_S15.nii')
s_light(True, 10, '../data/SL/', subjects, non_nan, header_fpath, output_fpath + 'AUC_S10.nii')
s_light(False, 0, '../data/SL/', subjects, non_nan, header_fpath, output_fpath + 'AUC_jointfit')
