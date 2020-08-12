import numpy as np
from scipy.io import loadmat
import nibabel as nib
import fnmatch, os, glob
from scipy.stats import zscore
from copy import deepcopy


class Dataset:

    def __init__(self, fpath):
        self.fpath = fpath
        self.data = []
        self.num_not_nan = []
        self.non_nan_mask = []

        # instead of retrieving affine and header data in pickles, just save as class attribute

    # add this func to pickles...


def get_repdata(fpath, subj_regex, condition_regex, file_ext='.nii', reps=6, subjs=[], roi_data=False):

    if len(subjs) > 0:
        subjects = deepcopy(subjs)
    else:
        subjects = [subjdir for subjdir in os.listdir(fpath) if fnmatch.fnmatch(subjdir, subj_regex)]

    D = []
    D_num_not_nan = []
    # D_not_nan_zs = []


    for rep in range(reps):

        #D_trial = np.zeros(loadmat(fpath + subjects[0] + subj_regex + '/' + glob.glob('*1*')[0])['rdata'].shape)
        D_trial = []
        D_not_nan = []

        for subj in range(len(subjects)):

            assert len(glob.glob(fpath + subjects[subj] + '/' + condition_regex + str(rep + 1) + file_ext)) == 1, "regex validator isn't working"

            fname = glob.glob(fpath + subjects[subj] + '/' + condition_regex + str(rep + 1) + file_ext)[0]

            if roi_data:
                from utils.pickles import hdf5_to_arr
                trial_z = hdf5_to_arr(fname)
                # trial_z = np.asarray(trial_z)

            else:

                trial_z = nib.load(fname).get_fdata()
                trial_z = (trial_z - np.mean(trial_z, axis=3)[:, :, :, np.newaxis])/(np.std(trial_z, axis=3)[:, :, :, np.newaxis])

            D_trial.append(trial_z)
            non_nan = np.array(~np.isnan(trial_z))
            D_not_nan.append(non_nan)

        D_not_nan = np.sum(D_not_nan, axis=0)
        #D_trial = np.array(D_trial).astype('float')
        #D_trial[D_trial == 0] = np.nan

        D_trial = np.nanmean(D_trial, axis=0)
        D_trial = (D_trial - np.nanmean(D_trial, axis=3)[:, :, :, np.newaxis])/(np.nanstd(D_trial, axis=3)[:, :, :, np.newaxis])

        # D_trial = zscore(D_trial, axis=1, ddof=1)
        D.append(D_trial.T)
        D_num_not_nan.append(D_not_nan.T)
        # D_not_nan_zs.append(~np.isnan(D_trial).T)


    # return D, D_num_not_nan
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

    # from utils.pickles import arr_to_hdf5
    #
    # subjects = [subjdir for subjdir in os.listdir(fpath) if fnmatch.fnmatch(subjdir, subj_regex)]
    #
    # roi_mask = nib.load(roi_mask_fpath).get_fdata()
    # n_rois = int(np.amax(roi_mask))
    #
    # z_scored_ds = []

    # get zscored data for each subject's repitions/trials #

    # for subj in range(len(subjects)):
    #
    #     reps_data = []
    #
    #     for rep in range(reps):
    #         print('subj = {0}'.format(subj))
    #         print('rep = {0}'.format(rep))
    #
    #         assert len(glob.glob(fpath + subjects[subj] + '/' + condition_regex + str(rep + 1) + '*' + file_ext)) == 1
    #
    #         fname = glob.glob(fpath + subjects[subj] + '/' + condition_regex + str(rep + 1) + '*' + file_ext)[0]
    #
    #         rep_data = nib.load(fname).get_fdata()
    #         rep_data = (rep_data - np.mean(rep_data, axis=3)[:, :, :, np.newaxis]) / (np.std(rep_data, axis=3)[:, :, :, np.newaxis])
    #         reps_data.append(rep_data.T)
    #
    #     z_scored_ds.append(reps_data)
    #
    # z_scored_ds = np.array(z_scored_ds)
    #
    #
    # for roi in range(1, n_rois + 1):
    #
    #     mask = roi_mask == roi
    #
    #     roi_data = z_scored_ds[:, :, :, mask]
    #
    #     arr_to_hdf5('dil_roi' + str(roi), data=roi_data)

    from utils.pickles import arr_to_hdf5

    subjects = [subjdir for subjdir in os.listdir(fpath) if fnmatch.fnmatch(subjdir, subj_regex)]

    roi_mask = nib.load(roi_mask_fpath).get_fdata()
    n_rois = int(np.amax(roi_mask))

    for roi in range(1, n_rois + 1):
        mask = roi_mask == roi
        z_scored_ds = []

        # get zscored data for each subject's repitions/trials #

        for subj in range(len(subjects)):

            reps_data = []

            for rep in range(reps):
                # print('subj = {0}'.format(subj))
                # print('rep = {0}'.format(rep))

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