import numpy as np
from brainiak.eventseg.event import EventSegment
from copy import deepcopy

from utils.labels import get_label
from num_events import get_numevents


def tj_fit(condition, data, nevents=7, avg_lasts=True):

    # assume data is nanmeaned average of all subjects... #
    label = get_label(condition)
    d = deepcopy(data)
    if avg_lasts:
        d = [d[0], np.nanmean(d[1:], axis=0)]
    d = np.asarray(d)
    nan_idxs = np.where(np.isnan(d))
    nan_idxs = list(set(nan_idxs[2]))

    d = np.delete(np.asarray(d), nan_idxs, axis=2)
    # get number of event segs for roi pass ONLY d[0] #
    # num_evsegs = get_numevents(d[0])

    # fit HMM #
    # ev_obj = EventSegment(num_evsegs).fit(d)
    # DS in dimensions: trial x tr x vox #
    ev_obj = EventSegment(nevents).fit(list(d))

    return ev_obj.segments_





