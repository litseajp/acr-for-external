import argparse
import os

import numpy as np
from sklearn.model_selection import KFold, train_test_split

from 01_utils.torch_utils import fix_seed


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

fix_seed(args.seed)


aud_files = os.listdir('02_audiofiles')
np.random.shuffle(aud_files)

train_files, test_files = train_test_split(aud_files, test_size=0.2)


all_feature_paths = []

for f in train_files:
    song_name = f.replace('.wav', '')
    feature_names = os.listdir(f'03_lowerfeatures/{song_name}')
    feature_paths = list(map(lambda x: f'{song_name}/{x}', feature_names))

    all_feature_paths += feature_paths

np.random.shuffle(all_feature_paths)


k_train_paths = []
k_val_paths = []
kf = KFold(n_splits=5)

for train_indices, val_indices in kf.split(all_feature_paths):
    k_train_paths.append([all_feature_paths[i] for i in train_indices])
    k_val_paths.append([all_feature_paths[i] for i in val_indices])


save_dir = '01_utils'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

np.savez(f'{save_dir}/split_indices.npz',
         train1=k_train_paths[0], val1=k_val_paths[0],
         train2=k_train_paths[1], val2=k_val_paths[1],
         train3=k_train_paths[2], val3=k_val_paths[2],
         train4=k_train_paths[3], val4=k_val_paths[3],
         train5=k_train_paths[4], val5=k_val_paths[4],
         test=test_files)
