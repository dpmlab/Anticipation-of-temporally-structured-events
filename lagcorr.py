import numpy as np
from utils import get_DTs, lag_pearsonr, save_nii, tj_fit

def lag_corr(dataset, roi_clusters, ev_conv, max_lag,
             header_fpath, save_prefix):
    """Lag correlation between HMM events and hand-annotated event data

    Refits an HMM within each cluster of roi_clusters, and computes a lag
    correlation between the derivative of the expected value of the event
    and ev_conv.

    Parameters
    ----------
    dataset : Dataset
        Data and mask for searchlight analysis
    roi_clusters : ndarray
        Mask of ROI clusters.
    ev_conv : ndarray
        Event boundaries convolved with the HRF
    max_lag : int
        Maximum lag to compute correlations for
    header_fpath : string
        File to use as a nifti template
    save_prefix : string
        Partial path for saving lag-0 maps

    Returns
    -------
    first_lagcorr : ndarray
        Lag correlation for first viewing (121 x 145 x 121 x (1+ 2*max_lag))
    lasts_lagcorr : ndarray
        Lag correlation for later viewings (121 x 145 x 121 x (1+ 2*max_lag))
    """

    vox_map_shape = (121, 145, 121, 1 + 2*max_lag)
    n_rois = int(np.max(roi_clusters))

    first_lagcorr = np.full(shape=vox_map_shape, fill_value=np.nan)
    lasts_lagcorr = np.full(shape=vox_map_shape, fill_value=np.nan)

    print('Computing lag correlations for ' + str(n_rois) + ' clusters')
    for cluster in range(1, n_rois + 1):
        mask = roi_clusters == cluster

        cluster_data = dataset.data[:, :, mask]
        segs = tj_fit(cluster_data)

        dts_first = get_DTs(segs[0])
        dts_lasts = get_DTs(segs[1])

        first_lagcorr[mask] = lag_pearsonr(dts_first, ev_conv[1:], max_lag)
        lasts_lagcorr[mask] = lag_pearsonr(dts_lasts, ev_conv[1:], max_lag)

    # Save lag-0 results
    save_nii(save_prefix + '_first.nii', header_fpath,
             first_lagcorr[:,:,:,max_lag])
    save_nii(save_prefix + '_lasts.nii', header_fpath,
             lasts_lagcorr[:,:,:,max_lag])
    save_nii(save_prefix + '_diff.nii', header_fpath,
             first_lagcorr[:,:,:,max_lag] - lasts_lagcorr[:,:,:,max_lag])

    return first_lagcorr, lasts_lagcorr
