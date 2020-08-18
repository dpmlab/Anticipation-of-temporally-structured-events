import numpy as np
from copy import deepcopy
from scipy.stats import norm, kstest, pearsonr


def get_AUCs(nevents, reps, segs):

    auc = [round(np.dot(segs[rep], np.arange(nevents)).sum(), 2) for rep in range(reps)]

    return np.asarray(auc)


def get_conf_int(original_samp, resamps, conf_per=95):

    resamps_sorted = deepcopy(resamps)
    resamps_sorted = np.sort(resamps_sorted, kind='mergesort')

    n_lessthan0 = np.count_nonzero(resamps_sorted < 0)
    idx_conf_min_val = int(((100 - conf_per) / 100) * len(resamps_sorted))
    conf_min_val = resamps_sorted[idx_conf_min_val]
    samp_5 = resamps_sorted[4]


    # if n_lessthan0 >= int(100 - conf_per):
    # if n_lessthan0 >= int(((100 - conf_per) / 100) * len(resamps_sorted)):
    #     return [np.nan, (np.nan, np.nan), n_lessthan0, conf_min_val, np.nan, np.nan, np.nan, np.round((n_lessthan0/len(resamps_sorted)), 6)]

    crit_val = ((100 - conf_per) / 2) / 100

    idx_low = int(len(resamps_sorted) * crit_val)
    idx_hi = int(len(resamps_sorted) - idx_low) - 1

    conf_int = (resamps_sorted[idx_low], resamps_sorted[idx_hi])

    pval = np.round((n_lessthan0/len(resamps_sorted)), 6)

    is_withinConf_Int = False
    if (original_samp > conf_int[0]) and (original_samp < conf_int[1]):
        is_withinConf_Int = True

    if n_lessthan0 == 0:
        return [is_withinConf_Int, conf_int, n_lessthan0, conf_min_val, np.mean(resamps_sorted[int(100 - conf_per):]),
                np.std(resamps_sorted[int(100 - conf_per):]), np.var(resamps_sorted[int(100 - conf_per):]),
                0.000001, samp_5]
    else:
        return [is_withinConf_Int, conf_int, n_lessthan0, conf_min_val, np.mean(resamps_sorted[int(100 - conf_per):]), np.std(resamps_sorted[int(100 - conf_per):]), np.var(resamps_sorted[int(100 - conf_per):]), np.round((n_lessthan0/len(resamps_sorted)), 6), samp_5]

def get_confval(resamps):

    resamps_sorted = deepcopy(resamps)
    resamps_sorted = np.sort(resamps_sorted, kind='mergesort')

    samp_5 = resamps_sorted[4]

    return samp_5


def get_percent_pos_neg(resamps):

    return sum(resamps > 0), sum(resamps < 0)


def get_pdf(original_samp, mean_resamps, std_resamps):
    return round(norm.pdf(original_samp, mean_resamps, std_resamps), 6)


def get_sf(original_samp, mean_resamps, std_resamps):
    return round(norm.sf(original_samp, mean_resamps, std_resamps), 6)

def get_CV_sf(mean_resamps, std_resamps):
    return round(norm.sf(mean_resamps/std_resamps), 6)

def get_ks_test(resamps, pval=.05):
    is_norm = kstest(resamps, 'norm', args=(np.mean(resamps), np.std(resamps)))[1] < pval
    return is_norm


def FDR_p(pvals):

    # Written by Chris Baldassano (git: cbaldassano), given permission to adapt into my code on 04/18/2019 #

    # Port of AFNI mri_fdrize.c
    assert np.all(pvals>=0) and np.all(pvals<=1)
    pvals[pvals < np.finfo(np.float_).eps] = np.finfo(np.float_).eps
    pvals[pvals == 1] = 1-np.finfo(np.float_).eps
    n = pvals.shape[0]

    qvals = np.zeros((n))
    sorted_ind = np.argsort(pvals)
    sorted_pvals = pvals[sorted_ind]
    qmin = 1.0
    for i in range(n-1,-1,-1):
        qval = (n * sorted_pvals[i])/(i+1)
        if qval > qmin:
            qval = qmin
        else:
            qmin = qval
        qvals[sorted_ind[i]] = qval

    # Estimate number of true positives m1 and adjust q
    if n >= 233:
        phist = np.histogram(pvals, bins=20, range=(0, 1))[0]
        sorted_phist = np.sort(phist[3:19])
        if np.sum(sorted_phist) >= 160:
            median4 = n - 20*np.dot(np.array([1, 2, 2, 1]), sorted_phist[6:10])/6
            median6 = n - 20*np.dot(np.array([1, 2, 2, 2, 2, 1]), sorted_phist[5:11])/10
            m1 = min(median4, median6)

            qfac = (n - m1)/n
            if qfac < 0.5:
                qfac = 0.25 + qfac**2
            qvals *= qfac

    return qvals


def lag_correlation(x, y, max_lags):

    assert max_lags < min(len(x), len(y)) / 2, "max_lags exceeds half the length of smallest array"

    assert len(x) == len(y), "array lengths are not equal"

    lag_corrs = np.full(1 + (max_lags * 2), np.nan)

    for idx in range(max_lags + 1):

        ## add correlations where y is ahead x ##
        lag_corrs[idx + max_lags] = pearsonr(x[:len(x) - idx], y[idx:len(y)])[0]

        ## add correlations where y is behind x ##
        lag_corrs[max_lags - idx] = pearsonr(x[idx:len(x)], y[:len(y) - idx])[0]

    return lag_corrs


def ev_annot_freq(ev_annots):

    frequencies = np.bincount(ev_annots)
    return np.array(frequencies[1:], dtype=np.float64)


def hrf_convolution(ev_annots_freq, n_participants, dts=np.arange(0, 15), pp=8.6, qq=0.547, TR=1.5, nTR=60):

    X = deepcopy(ev_annots_freq)
    X /= n_participants

    T = len(X)

    hrf = lambda dt, p, q: np.power(dt / (p * q), p) * np.exp(p - dt / q)

    X_conv = np.convolve(X, hrf(dts, pp, qq))[:T]


    return np.interp(np.linspace(0, (nTR - 1) * TR, nTR), np.arange(0, T), X_conv)


def get_DTs(ev_segs, nTR=60, n_events=7):

    evs = np.dot(ev_segs, np.arange(n_events))

    return [(evs[tr + 1] - evs[tr]) / ((tr + 1) - tr) for tr in range(nTR - 1)]

