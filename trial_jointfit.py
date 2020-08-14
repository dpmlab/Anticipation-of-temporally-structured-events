import numpy as np
from brainiak.eventseg.event import EventSegment
from copy import deepcopy

def tj_fit(data, n_events=7, avg_lasts=True):

    """

    Fits HMM to data and extracts event boundaries from repetitions.

    Assumes data is a nan-meaned (use numpy.nanmean) average of all subjects.

    :param data: array_like
        Data dimensions: Repetition x TR x Voxels

    :param n_events: int
        Specify number of boundaries to test.

    :param avg_lasts: bool, optional
        If true, HMM is fit to a) first viewing and b) the average of subsequent viewings.

    :return ev_obj.segments_: array_like
        Results of model fit
    """

    d = deepcopy(data)

    if avg_lasts:
        d = [d[0], np.nanmean(d[1:], axis=0)]
    d = np.asarray(d)

    nan_idxs = np.where(np.isnan(d))
    nan_idxs = list(set(nan_idxs[2]))

    d = np.delete(np.asarray(d), nan_idxs, axis=2)

    ev_obj = EventSegment(n_events).fit(list(d))

    return ev_obj.segments_





