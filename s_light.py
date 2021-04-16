import pickle
import numpy as np
from numpy.random import default_rng
from scipy.spatial.distance import cdist
from scipy.stats import norm, spearmanr
from utils import get_AUCs, tj_fit, save_nii, hyperalign, heldout_ll, FDR_p, \
                  get_DTs, ev_annot_freq, hrf_convolution, lag_pearsonr, \
                  nearest_peak


def get_s_lights(coords, stride=5, radius=5, min_vox=20):
    """Defines a grid of searchlights

    Defines a grid from 0 to the maximum coordinate in each dimension, with
    distance between grid points = stride. Each grid point is the center of a
    circular searchlight, which includes all coords within the defined
    radius. Searchlights with fewer than min_vox coordinates are discarded.

    Parameters
    ----------
    coords : ndarray
        V x 3 array, listing XYZ coordinates of all valid voxels
    stride : int
        Grid spacing
    radius : float
        Size of searchlight spheres
    min_vox : int
        Minimum number of voxels for a valid searchlight

    Returns
    -------
    list of ndarrays
        Each list element is the indices of coordinates in a searchlight
    """

    SL_allvox = []

    for x in range(0, np.max(coords, axis=0)[0] + stride, stride):
        for y in range(0, np.max(coords, axis=0)[1] + stride, stride):
            for z in range(0, np.max(coords, axis=0)[2] + stride, stride):
                dists = cdist(coords, np.array([[x, y, z]]))[:, 0]
                SL_vox = np.where(dists <= radius)[0]
                if len(SL_vox) >= min_vox:
                    SL_allvox.append(SL_vox)

    return SL_allvox

def optimal_events(data_list, subjects):
    """Find optimal number of events according to log-likelihood on first rep

    The event segmentation model is fit with varying number of events, and
    the optimal number of events is chosen based on held-out log-likelihood
    on the first repetition. The subjects are split into training and testing
    halves based on whether they were in the trw01 or trw02 group.

    Parameters
    ----------
    data_list : list of ndarrays
        List of Reps x TRs x Vox arrays for each subject
    subjects : list of strings
        Names of all subjects

    Returns
    -------
    int
        Number of events with highest log-likelihood
    """

    K_range = np.arange(2, 10)
    ll = np.zeros(len(K_range))
    split = np.array([('predtrw01' in s) for s in subjects])
    rep1 = np.array([d[0] for d in data_list])
    for i, K in enumerate(K_range):
        ll[i] = heldout_ll(rep1, K, split)
    return K_range[np.argmax(ll)]

def compile_optimal_events(pickle_path, non_nan_mask, SL_allvox,
                           header_fpath, save_path):
    """Create MNI map of optimal event numbers

    Parameters
    ----------
    pickle_path : string
        Filepath to where pickles were saved for each searchlight
    non_nan_mask : ndarray
        3d boolean mask of valid voxels
    SL_allvox : list of ndarrays
        List of voxel indices for each searchlight
    header_fpath : string
        Filepath of nii file with header to use as a template
    save_path : string
        Location of output directory
    """

    nSL = 5354

    sl_K = nSL*[None]
    for sl_i in range(nSL):
        pickle_fname = '%soptimal_events_%d.p' % (pickle_path, sl_i)
        sl_K[sl_i] = pickle.load(open(pickle_fname, 'rb'))

    K_vox3d = get_vox_map(sl_K, SL_allvox, non_nan_mask, return_q=False)
    save_nii(save_path + 'optimal_events.nii', header_fpath, K_vox3d)


def fit_HMM(data_list):
    """Hyperalign and fit HMM to data in one searchlight

    Parameters
    ----------
    data_list : list of ndarrays
        List of Reps x TRs x Vox arrays for each subject

    Returns
    -------
    list of ndarrays
        List of segmentations for each repetition
    """
    hyp_data = hyperalign(data_list)
    group_data = np.mean(hyp_data, axis=0)

    return tj_fit(group_data)

def compile_fit_HMM(pickle_path, non_nan_mask, SL_allvox,
                    header_fpath, save_path, opt_event):
    """Create MNI map of HMM fits and compute statistics

    Parameters
    ----------
    pickle_path : string
        Filepath to where pickles were saved for each searchlight
    non_nan_mask : ndarray
        3d boolean mask of valid voxels
    SL_allvox : list of ndarrays
        List of voxel indices for each searchlight
    header_fpath : string
        Filepath of nii file with header to use as a template
    save_path : string
        Location of output directory
    opt_event : ndarray
        3d volume, result of optimal_event analysis
    """

    nSL = 5354
    nPerm = 100
    TR = 1.5
    nEvents = 7
    max_lag = 10

    ev_conv = hrf_convolution(ev_annot_freq())
    lag_corr = nSL*[None]
    sl_AUCdiffs = nSL*[None]
    peak_shift = nSL*[None]

    # Load data from all searchlights
    for sl_i in range(nSL):
        sl_AUCdiffs[sl_i] = np.zeros((6-1, nPerm))
        lag_corr[sl_i] = np.zeros((6, 1 + 2*max_lag, nPerm))
        peak_shift[sl_i] = np.zeros(nPerm)

        pickle_fname = '%sfit_HMM_%d.p' % (pickle_path, sl_i)
        pick_data = pickle.load(open(pickle_fname, 'rb'))

        # Compute anticipation and shift in correlation with annotations
        for p in range(nPerm):
            seg = pick_data[p]
            AUC = get_AUCs(seg)
            sl_AUCdiffs[sl_i][:,p] = TR/(nEvents-1) * (AUC[1:]-AUC[0])
            peaks = np.zeros(6)
            for rep in range(6):
                sl_DT = get_DTs(seg[rep])
                lag_corr[sl_i][rep,:,p] = lag_pearsonr(sl_DT, ev_conv[1:],
                                                       max_lag)
                peaks[rep] = nearest_peak(lag_corr[sl_i][rep,:,p])
            peak_shift[sl_i][p] = TR*(peaks[1:].mean(0)-peaks[0])

        # Compute statistics for SLs for Figure 5
        if sl_i in [2614, 1479, 1054]:
            nBoot = 100
            bootstrap_rng = default_rng(0)
            boot_peak = np.zeros((nBoot, 6))
            for b in range(nBoot):
                ev_conv = hrf_convolution(ev_annot_freq(bootstrap_rng))
                for rep in range(6):
                    sl_DT = get_DTs(pick_data[0][rep])
                    boot_lag = lag_pearsonr(sl_DT, ev_conv[1:], max_lag)
                    boot_peak[b,rep] = nearest_peak(boot_lag)
            CI_init = TR*(max_lag - np.sort(boot_peak[:,0])[[5,95-1]])
            CI_rep = TR*(max_lag - np.sort(boot_peak[:,1:].mean(1))[[5,95-1]])

            print('%d: First Peak CI = %f, Rep Peak CI = %f' %
                  (sl_i, CI_init, CI_rep))

    # Create map of shifts in peak correlation with annotations
    pldiff, pldiff_q = get_vox_map(peak_shift, SL_allvox, non_nan_mask)
    save_nii(save_path + 'peaklagdiff.nii', header_fpath, pldiff)
    save_nii(save_path + 'peaklagdiff_q.nii', header_fpath, pldiff_q)


    # Create anticipation maps for each repetition and the average
    AUCdiff, AUCdiff_q = get_vox_map(sl_AUCdiffs, SL_allvox, non_nan_mask)
    for i in range(AUCdiff.shape[3]):
        save_nii(save_path + 'AUCdiff_' + str(i) + '.nii', header_fpath,
                 AUCdiff[:,:,:,i])
        save_nii(save_path + 'AUCdiff_' + str(i) + '_q.nii', header_fpath,
                 AUCdiff_q[:,:,:,i])

    for sl_i in range(nSL):
        sl_AUCdiffs[sl_i] = sl_AUCdiffs[sl_i].mean(0)
    AUCdiff, AUCdiff_q = get_vox_map(sl_AUCdiffs, SL_allvox, non_nan_mask)
    save_nii(save_path + 'AUCdiff_' + str(i) + '_mean.nii', header_fpath,
             AUCdiff)
    save_nii(save_path + 'AUCdiff_' + str(i) + '_mean_q.nii', header_fpath,
             AUCdiff_q)

    # Correlate anticipation with coordinates
    coords_nonnan = np.transpose(np.where(non_nan_mask))
    perm_maps = get_vox_map([sl[:,np.newaxis] for sl in sl_AUCdiffs],
                            SL_allvox, non_nan_mask, return_q = False)

    AUC_nonnan = perm_maps[non_nan_mask]
    spear = np.zeros((nPerm, 3))
    for p in range(nPerm):
        spear[p,:] = spearmanr(AUC_nonnan[:,p], coords_nonnan)[0][0,1:]
    print('Spearman corr w/coords (unmasked) ZYX=', spear[0,:])
    z = (spear[0,:]-spear[1:,:].mean(0))/np.std(spear[1:,:], axis=0)
    print('p vals=', norm.sf(z))

    qmask = AUCdiff_q[non_nan_mask] < 0.05
    coords_q05 = coords_nonnan[qmask,:]
    AUC_q05 = AUC_nonnan[qmask,:]
    spear = np.zeros((nPerm, 3))
    for p in range(nPerm):
        spear[p,:] = spearmanr(AUC_q05[:,p], coords_q05)[0][0,1:]
    print('Spearman corr w/coords (q<0.05 masked) ZYX=', spear[0,:])
    z = (spear[0,:]-spear[1:,:].mean(0))/np.std(spear[1:,:], axis=0)
    print('p vals=', norm.sf(z))


    # Correlate anticipation map and optimal event map
    K = opt_event
    K_nonnan = K[non_nan_mask]
    K_q05 = K_nonnan[qmask]
    K_spear = np.zeros(nPerm)
    for p in range(nPerm):
        K_spear[p] = spearmanr(AUC_q05[:,p], 90/K_q05)[0]
    print('Spearman corr w/K (q<0.05 masked) =', K_spear[0])
    z = (K_spear[0]-K_spear[1:].mean(0))/np.std(K_spear[1:])
    print('p val=',norm.sf(z))

def shift_corr(data_list, max_shift):
    """Compute cross-correlation between initial and repeated viewings

    Parameters
    ----------
    data_list : list of ndarrays
        List of Reps x TRs x Vox arrays for each subject
    max_shift : int
        Maximum lag between intial and repeated viewings

    Returns
    -------
    ndarray
        Array of 1 + 2*max_shift lag correlations, first value is correlation
        for initial viewing shifted earlier by max_shift timepoints
    """

    group_data = np.mean(data_list, axis=0).mean(2) # Rep x TR
    rep1 = group_data[0,:]
    rep2_6 = group_data[1:,:].mean(0)

    return lag_pearsonr(rep1, rep2_6, max_shift)

def compile_shift_corr(pickle_path, non_nan_mask, SL_allvox,
                       header_fpath, save_path):
    """Create map of peak of shift_corr

    Parameters
    ----------
    pickle_path : string
        Filepath to where pickles were saved for each searchlight
    non_nan_mask : ndarray
        3d boolean mask of valid voxels
    SL_allvox : list of ndarrays
        List of voxel indices for each searchlight
    header_fpath : string
        Filepath of nii file with header to use as a template
    save_path : string
        Location of output directory
    """

    nSL = 5354
    nPerm = 100
    TR = 1.5
    max_lag = 10

    corrshift = nSL * [None]
    for sl_i in range(nSL):
        corrshift[sl_i] = np.zeros(nPerm)
        pickle_fname = '%sshift_corr_%d.p' % (pickle_path, sl_i)
        pick_data = pickle.load(open(pickle_fname, 'rb'))

        for p in range(nPerm):
            lag_corr = pick_data[p][:]
            corrshift[sl_i][p] = TR*(max_lag - nearest_peak(lag_corr))

        cs, cs_q = get_vox_map(corrshift, SL_allvox, non_nan_mask)
        save_nii(save_path + 'shift_corr.nii', header_fpath, cs)
        save_nii(save_path + 'shift_corr_q.nii', header_fpath, cs_q)

def get_vox_map(SL_results, SL_voxels, non_nan_mask, return_q=True):
    """Projects searchlight results to voxel maps.

    Parameters
    ----------
    SL_results: list of ndarrays
        List of SL results, each of length nPerm or shape nMaps x nPerm
    SL_voxels: list
        Voxel information from searchlight analysis
    non_nan_mask: ndarray
        3d boolean mask indicating elements containing data
    return_q : boolean
        Whether to compute and return FDR-corrected p values

    Returns
    -------
    ndarray
        Map of values in each voxel

    ndarray
        Map of q values for each voxel (if return_q=True)
    """

    coords = np.transpose(np.where(non_nan_mask))
    nVox = coords.shape[0]
    if SL_results[0].ndim == 1:
        nMaps = 1
        nPerm = len(SL_results[0])
    else:
        nMaps = SL_results[0].shape[0]
        nPerm = SL_results[0].shape[1]

    voxel_maps = np.zeros((nMaps, nPerm, nVox))
    voxel_SLcount = np.zeros(nVox)

    for idx, sl in enumerate(SL_voxels):
        if nMaps == 1:
            voxel_maps[0,:,sl] += SL_results[idx]
        else:
            for m in range(nMaps):
                voxel_maps[m,:,sl] += SL_results[idx][m, :]
        voxel_SLcount[sl] += 1

    nz_vox = voxel_SLcount > 0
    voxel_maps[:, :, nz_vox] = voxel_maps[:, :, nz_vox] / voxel_SLcount[nz_vox]
    voxel_maps[:, :, ~nz_vox] = np.nan

    vox3d = np.full(non_nan_mask.shape + (nMaps,), np.nan)
    vox3d[non_nan_mask,:] = voxel_maps[:,0,:].T

    if not return_q:
        return vox3d.squeeze()

    null_means = voxel_maps[:, 1:, nz_vox].mean(1)
    null_stds = np.std(voxel_maps[:, 1:, nz_vox], axis=1)

    z = (voxel_maps[:, 0, nz_vox] - null_means)/null_stds
    p = norm.sf(z)
    q = np.zeros(p.shape)
    for m in range(nMaps):
        q[m,:] = FDR_p(p[m,:])

    z3d = np.full(non_nan_mask.shape + (nMaps,), np.nan)
    z3d[non_nan_mask,:] = z.T
    q3d = np.full(non_nan_mask.shape + (nMaps,), np.nan)
    q3d[non_nan_mask,:] = q.T

    return vox3d.squeeze(), q3d.squeeze()
