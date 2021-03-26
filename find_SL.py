import pickle
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import matplotlib.ticker as plticker
import matplotlib

nSL = 5354

SL_allvox = pickle.load(open('../data/SL/SL_allvox.p', 'rb'))
non_nan_mask = nib.load('../data/valid_vox.nii').get_fdata().T > 0
coords_nonnan = np.transpose(np.where(non_nan_mask))  # ZYX

q = nib.load('../outputs/AUC_S10_jointfit_mean_q.nii').get_fdata().T
q = q[non_nan_mask]
sigcoords = np.where((q>0)*(q<0.05))[0]

AUC, DT = pickle.load(open('../outputs/AUC_S10_jointfit.p', 'rb'))

ordered = np.where([ (np.min(AUC[i][1:]) > AUC[i][0]) and (np.min(AUC[i][2:] > AUC[i][1])) for i in range(nSL)])[0]
sig = np.where([ (len(np.intersect1d(SL_allvox[i], sigcoords))/len(SL_allvox[i]) > 0.5) for i in range(nSL)])[0]

valid_SL = np.intersect1d(ordered, sig)

SL_y = np.array([coords_nonnan[vox,1].mean(0) for vox in SL_allvox])

dt_std = np.array([np.std(dt, axis=1).mean() for dt in DT])

pred = np.array([x[1:].mean() - x[0] for x in AUC])
exp = [np.concatenate((np.zeros((6,1)), np.cumsum(dt, axis=1)), axis=1) for dt in DT]
pred_std = np.array([np.std(e[1:,:].mean(0)-e[0,:])for e in exp])
min_exp = np.array([(np.argmin(e, axis=0)==0).mean() for e in exp])


# Posterior
valid_SL1 = reduce(np.intersect1d, (valid_SL, np.where(SL_y <= 40)[0], np.where(dt_std > 0.1)[0], np.where(pred_std < 0.25)[0], np.where(min_exp > 0.5)[0]))
SL1 = valid_SL1[np.argmax(pred[valid_SL1])]
print(pred[SL1])

# Mid
valid_SL2 = reduce(np.intersect1d, (valid_SL, np.where((SL_y > 40)*(SL_y<100))[0], np.where(dt_std > 0.05)[0], np.where(pred_std < 0.25)[0], np.where(min_exp > 0.5)[0]))
SL2 = valid_SL2[np.argmax(pred[valid_SL2])]
print(pred[SL2])

# Anterior
valid_SL3 = reduce(np.intersect1d, (valid_SL, np.where(SL_y>=100)[0], np.where(dt_std > 0.03)[0], np.where(pred_std < 0.5)[0], np.where(min_exp > 0.5)[0]))
SL3 = valid_SL3[np.argmax(pred[valid_SL3])]
print(pred[SL3])
quit()

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
colors = ['#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#b10026']
for sl in [SL1, SL2, SL3]:
    plt.figure(figsize=(3,3))
    for rep in range(6):
        plt.plot(np.arange(60)*1.5,exp[sl][rep,:]+1, colors[rep])

    plt.legend(np.arange(6))
    plt.xlabel('Time in movie (sec)')
    plt.ylabel('Events')
    plt.xlim(0,90)
    plt.gca().xaxis.set_major_locator(plticker.MultipleLocator(base=30))
    plt.ylim(0,8)
    plt.yticks(np.arange(1,8))
    plt.tight_layout()
    plt.title(str(sl))
    plt.savefig('../outputs/' + str(sl)+'.png', dpi=200)
    plt.savefig('../outputs/' + str(sl)+'.svg')
    #plt.show()
