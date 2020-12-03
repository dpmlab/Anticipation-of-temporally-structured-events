import glob
import numpy as np
import nibabel as nib


class Dataset:

    def __init__(self, fpath, subjects, condition_regex, min_subjs=15):

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
