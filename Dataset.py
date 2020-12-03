import glob
import numpy as np
import nibabel as nib


class Dataset:

    def __init__(self):
        self.data = None
        self.non_nan_mask = None

    def load_data(self, fpath, subjects, condition_regex, min_subjs=15):
        """Loads data files matching a regular expression

        Saves z-scored data, per each repetition, averaged across subjects for
        a specific condition ('Intact', 'Scrambled-Fixed', etc.) in self.data
        and the counts of valid subjects at each voxel into self.num_not_nan.

        Parameters
        ----------
        fpath : string
            Path to data directory
        subjects : list
            List of subjects to load
        condition_regex : string
            Regular expression identifying a specific condition (e.g. 'Intact')
        min_subjs : int, optional
            Minimum number of subjects for a valid voxel
        """

        D = []
        D_num_not_nan = []

        MNI_mask_path = 'MNI152_T1_brain_resample.nii'
        print("Loading data", end='', flush=True)
        for rep in range(6):
            D_rep = np.zeros((121, 145, 121, 60))
            D_not_nan = np.zeros((121, 145, 121))

            for subj in subjects:
                print('.', end='', flush=True)
                fname = glob.glob(fpath + subj + '/' +
                                  condition_regex + str(rep + 1) + '.nii')

                assert len(fname) == 1, \
                       "More than one file found for subject " + subj

                rep_z = nib.load(fname[0]).get_fdata()
                nnan = ~np.all(rep_z == 0, axis=3)
                rep_mean = np.mean(rep_z[nnan], axis=1, keepdims=True)
                rep_std = np.std(rep_z[nnan], axis=1, keepdims=True)
                rep_z[nnan] = (rep_z[nnan] - rep_mean)/rep_std

                D_rep[nnan] += rep_z[nnan]
                D_not_nan[nnan] += 1

            nnan = D_not_nan > 0
            D_rep[nnan] = D_rep[nnan]/D_not_nan[nnan][:, np.newaxis]
            D_rep_mean = np.mean(D_rep[nnan], axis=1, keepdims=True)
            D_rep_std = np.std(D_rep[nnan], axis=1, keepdims=True)
            D_rep[nnan] = (D_rep[nnan] - D_rep_mean)/D_rep_std

            D.append(D_rep.T)
            D_num_not_nan.append(D_not_nan.T)
        print(' ')

        self.data = np.asarray(D)
        non_nan_mask = np.min(D_num_not_nan, axis=0) # Min across reps
        MNI_mask = nib.load(MNI_mask_path).get_fdata().astype(bool).T
        self.non_nan_mask = (non_nan_mask > (min_subjs - 1)) * MNI_mask

    def load_rois(self, roipath, subjects):
        """Loads data files previously saved by save_rois

        Parameters
        ----------
        roipath : string
            Directory with roi npy files
        subjects : list
            List of subjects to load
        """
        print("Loading rois", end='', flush=True)

        roi_mask = np.load(roipath + 'roi_mask.npy')
        self.non_nan_mask = roi_mask

        mean_data = np.zeros((6, 60, roi_mask.sum()))
        num_subj = np.zeros((6, roi_mask.sum()))
        for subj in subjects:
            print('.', end='', flush=True)
            roi_data = np.load(roipath + subj.split('/')[-1] + '.npy')
            for rep in range(6):
                nnan = ~np.all(roi_data[rep] == 0, axis=0)
                mean_data[rep][:, nnan] += roi_data[rep][:, nnan]
                num_subj[rep][nnan] += 1
        mean_data = mean_data / num_subj[:, np.newaxis, :]

        for rep in range(6):
            rep_mean = np.mean(mean_data[rep], axis=0, keepdims=True)
            rep_std = np.std(mean_data[rep], axis=0, keepdims=True)
            mean_data[rep] = (mean_data[rep] - rep_mean)/rep_std

        self.data = np.full((6, 60, 121, 145, 121), np.nan)
        self.data[:,:,roi_mask] = mean_data


def save_rois(fpath, subjects, condition_regex, roi_mask, savepath):
    """Save a data from a specified mask for each subject

    For ROI-based analyses, this function can be used to save small
    npy files that can be loaded by load_rois() rather than loading
    the full nifti files and then masking.

    Parameters
    ----------
    fpath : string
        Path to data directory
    subjects : list
        List of subjects to load
    condition_regex : string
        Regular expression identifying a specific condition (e.g. 'Intact')
    roi_mask : ndarray
        3d boolean array of which voxels to include
    savepath : string
        Directory to save generated files
    """

    print("Loading data", end='', flush=True)
    np.save(savepath + 'roi_mask', roi_mask)
    for subj in subjects:
        roi_data = np.zeros((6, 60, roi_mask.sum()))
        for rep in range(6):
            print('.', end='', flush=True)
            fname = glob.glob(fpath + subj + '/' +
                              condition_regex + str(rep + 1) + '.nii')

            rep_z = nib.load(fname[0]).get_fdata()[roi_mask.T]
            nnan = ~np.all(rep_z == 0, axis=1)
            rep_mean = np.mean(rep_z[nnan], axis=1, keepdims=True)
            rep_std = np.std(rep_z[nnan], axis=1, keepdims=True)
            rep_z[nnan] = (rep_z[nnan] - rep_mean)/rep_std

            roi_data[rep][:,nnan] = rep_z[nnan].T
        np.save(savepath + subj.split('/')[-1], roi_data)
