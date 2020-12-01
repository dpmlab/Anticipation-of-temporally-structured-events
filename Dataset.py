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
        reps = 6
        for rep in range(reps):
            D_rep = []
            D_not_nan = []

            for subj in subjects:

                fname = glob.glob(fpath + subj + '/' +
                                  condition_regex + str(rep + 1) + '.nii')[0]

                assert len(fname) == 1, \
                       "More than one file found for subject " + subj

                rep_z = nib.load(fname).get_fdata()
                rep_z = ((rep_z - np.mean(rep_z, axis=3, keepdims=True))
                         /np.std(rep_z, axis=3, keepdims=True))

                D_rep.append(rep_z)
                non_nan = np.array(~np.isnan(rep_z))
                D_not_nan.append(non_nan)

            D_not_nan = np.sum(D_not_nan, axis=0)
            D_rep = np.nanmean(D_rep, axis=0)
            D_rep = ((D_rep - np.nanmean(D_rep, axis=3, keepdims=True))
                     /np.nanstd(D_rep, axis=3, keepdims=True))

            D.append(D_rep.T)
            D_num_not_nan.append(D_not_nan.T)

        self.data = D
        non_nan_mask = np.min(D_num_not_nan, axis=0) # Min across reps
        MNI_mask = nib.load(MNI_mask_path).get_fdata().astype(bool).T
        self.non_nan_mask = (non_nan_mask > (min_subjs - 1)) * MNI_mask
