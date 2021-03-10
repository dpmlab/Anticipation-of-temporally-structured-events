import glob
import pickle

pickle_path = '../outputs/perm/pickles/'

pickles = glob.glob(pickle_path + '*.p')
for i, pick in enumerate(pickles):
    pick_parts = pick.split('_')
    analysis_i = int(pick_parts[1])
    sl_i = int(pick_parts[2])

    if analysis_i == 3:
        try:
            pick_data = pickle.load(open(pick, 'rb'))
        except:
            print(str(sl_i) + ' ', end='')
print(' ')
