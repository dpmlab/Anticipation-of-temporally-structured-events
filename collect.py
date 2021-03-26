import glob
from tqdm import tqdm
import pickle
import numpy as np
from utils import get_DTs

nSL = 5354
TR = 1.5
nTR = 60
nEvents = 7

AUC = nSL * [None]
DT = nSL * [None]
for i, pick in tqdm(enumerate(glob.glob('../outputs/perm/pickles/_5_*.p'))):
    pick_parts = pick.split('_')
    analysis_i = int(pick_parts[1])
    sl_i = int(pick_parts[2])

    with open (pick, 'rb') as fp:
        pick_data = pickle.load(fp)

        AUC[sl_i] = (TR/(nEvents-1) * pick_data[0][0])
        DT[sl_i] = np.zeros((6,nTR-1))
        for rep in range(6):
            DT[sl_i][rep,:] = get_DTs(pick_data[1][0][rep])

pickle.dump((AUC, DT), open('../outputs/AUC_S10_jointfit.p', 'wb'))
