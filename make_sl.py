from Dataset import save_s_lights
import nibabel as nib

non_nan = nib.load('../code_preprint/valid_vox.nii').get_fdata().T > 0
save_s_lights('../data/', non_nan, '../data/SL/')