import argparse
import os
import sys
import warnings

import librosa
import numpy as np
import torch
import yaml

from 01_utils.torch_utils import recomposition_c170
from 04_BTCmodel.model import BTC_model
from 04_CRNNmodel.model import CRNN_model


warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['CRNN', 'BTC'], default='CRNN')
parser.add_argument('--path', required=True)
args = parser.parse_args()

if torch.cuda.is_available():
    print('GPU Check: OK')
    device = 'cuda'
else:
    print('GPU Check: NG')
    sys.exit()

if not os.path.exists(args.path):
    print('ERROR: The file does not exist.')
    sys.exit()

with open('01_utils/feature_config.yaml', 'r') as conf:
    feature_config = yaml.safe_load(conf)

with open(f'04_{args.model}model/model_config.yaml', 'r') as conf:
    model_config = yaml.safe_load(conf)

if args.model == 'CRNN':
    model = CRNN_model(config=model_config['model']).to(device)
elif args.model == 'BTC':
    model = BTC_model(config=model_config['model']).to(device)
else:
    raise NotImplementedError


sr = feature_config['sampling_rate']
freq_bins = feature_config['freq_bins']
time_bins = feature_config['time_bins']
hop_length = feature_config['hop_length']
bins_per_octave = feature_config['bins_per_octave']
batch_size = model_config['experiment']['batch_size']

model.load_state_dict(torch.load(f'04_{args.model}model/saved_models/model_1.pth'))

std_data = np.load('01_utils/standardization/std_1.npz')
mean = torch.tensor(std_data['mean'])
std = torch.tensor(std_data['std'])


y, _ = librosa.load(args.path)

cqt = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=freq_bins, bins_per_octave=bins_per_octave)
cqt = np.log(np.abs(cqt) + 1e-6)

pad_len = time_bins - (cqt.shape[1] % time_bins)
cqt = np.concatenate([cqt, np.full((freq_bins, pad_len), np.min(cqt))], axis=1)

clip_n = cqt.shape[1] // time_bins
features = np.full((batch_size, freq_bins, time_bins), np.min(cqt))

for i in range(clip_n):
    features[i] = cqt[:, i*time_bins:i*time_bins+time_bins]

features = torch.tensor(features, dtype=torch.float)


chord_seq = []

with torch.no_grad():
    model.eval()

    features = features.to(device)
    features = features.permute(0, 2, 1)
    features = (features - mean) / std

    outputs = model(features)
    _, prediction = torch.max(outputs, 2)

    for i in range(clip_n):
        chord_seq += [recomposition_c170(note_n) for note_n in prediction[i]]


cur_note = chord_seq[0]
start_fr = 0

labdir = '09_result'

if not os.path.exists(labdir):
    os.makedirs(labdir)

lab_name = args.path.split('/')[-1].replace('.wav', '.lab')
est_lab = f'{labdir}/{lab_name}'

with open(est_lab, 'w') as f:
    for fr, nt in enumerate(chord_seq[1:]):
        if cur_note != nt:
            f.write(str(round(librosa.frames_to_time(start_fr, hop_length=hop_length), 6))+' '+str(
                round(librosa.frames_to_time(fr, hop_length=hop_length), 6))+' '+cur_note+'\n')
            cur_note = nt
            start_fr = fr

    f.write((' ').join([str(round(librosa.frames_to_time(start_fr, hop_length=hop_length), 6)),
                        str(round(librosa.frames_to_time(len(chord_seq)-1, hop_length=hop_length), 6)),
                        cur_note]))


print('Inference Finished.')
