import numpy as np
import nibabel as nib
import fnmatch, os, glob
from copy import deepcopy
from utils.pickles import arr_to_hdf5, hdf5_to_arr


class Dataset:

    def __init__(self, fpath):
        self.fpath = fpath
        self.data = []
        self.num_not_nan = []
        self.non_nan_mask = []


def get_repdata(fpath, subj_regex, condition_regex, file_ext='.nii', reps=6, subjs=[], roi_data=False, save_nonnanmask=False):

    """

    :param fpath: string

    :param subj_regex: string

    :param condition_regex: string

    :param file_ext: string, optional

    :param reps: int, optional

    :param subjs: list, optional

    :param roi_data: boolean, optional

    :param save_nonnanmask: boolean, optional

    :return D: array_like

    """

    if len(subjs) > 0:
        subjects = deepcopy(subjs)
    else:
        subjects = [subjdir for subjdir in os.listdir(fpath) if fnmatch.fnmatch(subjdir, subj_regex)]

    D = []
    D_num_not_nan = []


    for rep in range(reps):

        D_rep = []
        D_not_nan = []

        for subj in range(len(subjects)):

            fname = glob.glob(fpath + subjects[subj] + '/' + condition_regex + str(rep + 1) + file_ext)[0]

            assert len(fname) == 1, "More than one file found for subject " + subjects[subj]

            if roi_data:
                rep_z = hdf5_to_arr(fname)

            else:

                rep_z = nib.load(fname).get_fdata()
                rep_z = (rep_z - np.mean(rep_z, axis=3)[:, :, :, np.newaxis])/(np.std(rep_z, axis=3)[:, :, :, np.newaxis])

            D_rep.append(rep_z)
            non_nan = np.array(~np.isnan(rep_z))
            D_not_nan.append(non_nan)

        D_not_nan = np.sum(D_not_nan, axis=0)

        D_rep = np.nanmean(D_rep, axis=0)
        D_rep = (D_rep - np.nanmean(D_rep, axis=3)[:, :, :, np.newaxis])/(np.nanstd(D_rep, axis=3)[:, :, :, np.newaxis])

        D.append(D_rep.T)
        D_num_not_nan.append(D_not_nan.T)

    if save_nonnanmask:
        np.save('non_nan_mask.npy', D_num_not_nan)

    return D

def get_maskdata(fpath):

    return np.asarray(nib.load(fpath).get_fdata().astype(bool)).T

def get_non_nan_mask(non_nan_count, mask, min_subjs=15):

    non_nan_mask = np.asarray(deepcopy(non_nan_count))

    # get min number of subject data per voxel over repititions and then again over TRs #
    non_nan_mask = np.min(np.min(non_nan_mask, axis=0), axis=0)

    return (non_nan_mask > (min_subjs - 1)) * mask

def get_data_pval(voxel_data, p_vals, p=.05):

    masked_data = np.asarray(deepcopy(voxel_data))

    return masked_data * (p_vals <= p)


def get_slight_clusters(voxel_data):

    from scipy.ndimage.measurements import label

    voxel_data[np.isnan(voxel_data)] = 0

    return label(voxel_data)


def get_rois(fpath, subj_regex, condition_regex, roi_mask_fpath, file_ext='.nii', reps=6):

    subjects = [subjdir for subjdir in os.listdir(fpath) if fnmatch.fnmatch(subjdir, subj_regex)]

    roi_mask = nib.load(roi_mask_fpath).get_fdata()
    n_rois = int(np.amax(roi_mask))

    for roi in range(1, n_rois + 1):
        mask = roi_mask == roi
        z_scored_ds = []

        for subj in range(len(subjects)):

            reps_data = []

            for rep in range(reps):
                assert len(
                    glob.glob(fpath + subjects[subj] + '/' + condition_regex + str(rep + 1) + '*' + file_ext)) == 1

                fname = glob.glob(fpath + subjects[subj] + '/' + condition_regex + str(rep + 1) + '*' + file_ext)[0]

                rep_data = nib.load(fname).get_fdata()

                rep_data = rep_data[mask.T, :]  # roi voxels only

                rep_data = (rep_data - np.mean(rep_data, axis=1)[:, np.newaxis]) / (
                np.std(rep_data, axis=1)[:, np.newaxis])
                reps_data.append(rep_data.T)

            z_scored_ds.append(reps_data)

        roi_data = np.array(z_scored_ds)
        arr_to_hdf5('dil_roi' + str(roi), data=roi_data)


def get_roidata(roi, subj_idxs, fpath_regex='dil_roi'):

    assert len(glob.glob(fpath_regex + str(roi) + '.*')) == 1

    fname = glob.glob(fpath_regex + str(roi) + '.*')[0]

    from utils.pickles import hdf5_to_arr

    D = np.array(hdf5_to_arr(fname))
    D = D[subj_idxs, :]
    D = np.nanmean(D, axis=0)
    D = (D - np.nanmean(D, axis=1)[:, np.newaxis]) / (np.nanstd(D, axis=1)[:, np.newaxis])

    return D