import numpy as np
import nibabel as nib
import h5py
import os, fnmatch


def npz_to_array(f_path):
    d_in = np.load(f_path)
    d_in.files
    return [d_in['arr_' + str(elem)] for elem in range(len(d_in.files))]

def arr_to_nii(new_fpath, fpath_nii, data):

    img = nib.load(fpath_nii)
    new_img = nib.Nifti1Image(data, img.affine, img.header)
    #temp_img = nib.Nifti1Image(array, np.eye(eye), header=head)
    #nib.save(nib.Nifti1Image(temp_img, np.eye(eye)), fpath)
    nib.save(new_img, new_fpath)

    # fpath = 'MNI152_T1_brain_resample.nii'
    # temp_nan = np.asarray(hdf5_to_arr('dset_int_nan.h5'))
    # img = nib.load(fpath)
    # new_img = nib.Nifti1Image(temp_nan, img.affine, img.header)
    # nib.save(new_img, 'temp_nib.nii')


def arr_to_hdf5(fpath, data):

    hdf_obj = h5py.File(fpath + '.h5', 'w')

    for idx, arr in enumerate(data):
        hdf_obj.create_dataset(str(idx), data=arr)

    hdf_obj.close()


def hdf5_to_arr(fpath):

    hdf_obj = h5py.File(fpath, 'r')
    data = [i[:] for i in hdf_obj.values()]
    hdf_obj.close()

    return data


def to_npz(self, out_path):
    np.savez(out_path, self.data)


def split_nii(orig_file, out_path, fpath_nii, transpose=False):

    orig_data = nib.load(orig_file).get_fdata()

    for f_out in range(orig_data.shape[0]):
        if transpose:
            arr_to_nii(out_path + '_' + str(f_out + 1) + '.nii', fpath_nii, orig_data[f_out].T)
            continue
        else:
            arr_to_nii(out_path + '_' + str(f_out + 1) + '.nii', fpath_nii, orig_data[f_out])


def diff_int_sfix(int_regex, sfix_regex, n_files, res_dir, out_path, header_fpath):

    for n_file in range(1, n_files + 1):

        int_file = [file for file in os.listdir(res_dir) if fnmatch.fnmatch(file, int_regex + str(n_file) + '.nii')]
        sfix_file = [file for file in os.listdir(res_dir) if fnmatch.fnmatch(file, sfix_regex + str(n_file) + '.nii')]

        assert len(int_file) == 1
        assert len(int_file) == len(sfix_file)

        arr_to_nii(out_path + '_' + str(n_file) + '.nii', header_fpath, np.asarray(nib.load(res_dir + int_file[0]).get_fdata()) - np.asarray(nib.load(res_dir + sfix_file[0]).get_fdata()))


def means_bstraps(g1_regex, g2_regex, n_files, res_dir, outpath, header_fpath):

    for n_file in range(1, n_files + 1):

        g1 = [file for file in os.listdir(res_dir) if fnmatch.fnmatch(file, g1_regex + str(n_file) + '.nii')]
        g2 = [file for file in os.listdir(res_dir) if fnmatch.fnmatch(file, g2_regex + str(n_file) + '.nii')]

        assert len(g1) == 1
        assert len(g2) == len(g1)

        avg = np.asarray(nib.load(res_dir + g1[0]).get_fdata())
        avg += np.asarray(nib.load(res_dir + g2[0]).get_fdata())
        avg /= 2

        arr_to_nii(outpath + '_' + str(n_file) + '.nii', header_fpath, avg)