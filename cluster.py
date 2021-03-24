import nibabel as nib
import numpy as np
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse.csgraph import connected_components
from scipy.ndimage.measurements import label
from utils import save_nii

data_fpath = '../data/'
header_fpath = data_fpath + 'header.nii'

d = nib.load('../outputs/AUC_S10_jointfit_mean.nii').get_fdata().T
q = nib.load('../outputs/AUC_S10_jointfit_mean_q.nii').get_fdata().T
mask = (q > 0) * (q < 0.05)

# First find the main cluster
clusters = label(mask)[0]
values, counts = np.unique(clusters, return_counts=True)
largest_clust = clusters == values[np.argmax(counts[1:])+1]
largest_clust_ind = largest_clust.reshape(-1)
connectivity = grid_to_graph(d.shape[0], d.shape[1], d.shape[2]).tocsr()
connectivity = connectivity[largest_clust_ind][:,largest_clust_ind]

ward = AgglomerativeClustering(n_clusters=20, linkage='ward',
                               connectivity=connectivity)

coords = np.transpose(np.where(largest_clust))
X = np.concatenate((d[largest_clust][:, np.newaxis], 0.1*coords), axis=1)
ward.fit(X)
c = ward.labels_

c3d = np.full(d.shape, np.nan)
c3d[largest_clust] = c
save_nii('../outputs/AUC_S10_jointfit_mean_ward.nii', header_fpath, c3d)