from copy import deepcopy
import numpy as np
from brainiak.eventseg.event import EventSegment

def get_numevents(data, window=5, min_ev=2, max_ev=15):
    d = deepcopy(data)
    win_v_across = []

    # change this to Pearson's coeff #
    win_corr = [np.corrcoef(d[t], d[t + window])[1, 0] for t in range(d.shape[0] - window)]

    # using leave-one-out bootstrapping here...
    for idx in range(len(d)):

        boot_d = np.mean(d[:idx] + d[idx + 1:], axis=0)

        for num_events in range(min_ev, max_ev + 1):

            ev_obj = EventSegment(num_events).fit(boot_d)
            segments = ev_obj.segments_[0]
            ev = np.argmax(segments, axis=1)

            same_event = [ev[tr] == ev[tr + window] for tr in range(ev.shape[0] - window)]
            within = np.mean([win_corr[tr] for tr, ev in enumerate(same_event) if ev])
            across = np.mean([win_corr[tr] for tr, ev in enumerate(same_event) if not ev])

            if idx == 0:
                win_v_across.append([])
            win_v_across[num_events - 2].append(np.subtract(within, across))

    win_v_across = list(map((lambda x: np.mean(x)), win_v_across))

    return np.argmax(win_v_across) + 2