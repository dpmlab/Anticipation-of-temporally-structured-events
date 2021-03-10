from os import path
import sys
import numpy as np
import glob

pickle_path = '../outputs/perm/pickles/'
max_sl = int(sys.argv[1])+1

complete = np.zeros((max_sl, 5), dtype=bool)
pickles = glob.glob(pickle_path + '*.p')
for i, pick in enumerate(pickles):
    pick_parts = pick.split('_')
    analysis_i = int(pick_parts[1])
    sl_i = int(pick_parts[2])

    complete[sl_i, analysis_i] = True

for i in range(5):
	print('Analysis',i, ':',complete[:,i].sum(), '/', max_sl)

print(' ')
for i in range(max_sl):
	if np.any(~complete[i,:]):
		print(str(i)+',', end='')

print(' ')