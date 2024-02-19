import argparse
import os
import sys
import warnings
from statistics import mean as calc_mean

import librosa
import numpy as np
import torch
import yaml
from tqdm import tqdm

from 01_utils.torch_utils import evaluate_chordfile, recomposition_c170
from 04_BTCmodel.model import BTC_model
from 04_CRNNmodel.model import CRNN_model


warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['CRNN', 'BTC'], default='CRNN')
args = parser.parse_args()

if torch.cuda.is_available():
    print('GPU Check: OK')
    device = 'cuda'
else:
    print('GPU Check: NG')
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


kf = np.load('01_utils/split_indices.npz')
test_files = kf['test']

sr = feature_config['sampling_rate']
freq_bins = feature_config['freq_bins']
time_bins = feature_config['time_bins']
hop_length = feature_config['hop_length']
bins_per_octave = feature_config['bins_per_octave']
batch_size = model_config['experiment']['batch_size']


all_scores = [[] for _ in range(7)]
all_scores_append = [all_scores[i].append for i in range(7)]

for index in range(1, 6):
    model.load_state_dict(torch.load(f'04_{args.model}model/saved_models/model_{index}.pth'))

    std_data = np.load(f'01_utils/standardization/std_{index}.npz')
    mean = torch.tensor(std_data['mean'])
    std = torch.tensor(std_data['std'])

    scores = [[] for _ in range(7)]
    scores_append = [scores[i].append for i in range(7)]

    for audio_file in tqdm(test_files):
        y, _ = librosa.load(f'02_audiofiles/{audio_file}')

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

        labdir = f'08_temp/est_lab/idx_{index}'

        if not os.path.exists(labdir):
            os.makedirs(labdir)

        lab_name = audio_file.replace('.wav', '.lab')
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

        ref_lab = f'02_chordfiles/{lab_name}'

        score_list = evaluate_chordfile(ref_lab, est_lab)

        for i in range(7):
            scores_append[i](score_list[i])

    print(f'root: {calc_mean(scores[0]):.2%} thirds: {calc_mean(scores[1]):.2%} '
          f'majmin: {calc_mean(scores[2]):.2%} triads: {calc_mean(scores[3]):.2%} '
          f'sevenths: {calc_mean(scores[4]):.2%} tetrads: {calc_mean(scores[5]):.2%} '
          f'mirex: {calc_mean(scores[6]):.2%}')

    for i in range(7):
        all_scores_append[i](calc_mean(scores[i]))

print(f'root: {calc_mean(all_scores[0]):.2%} thirds: {calc_mean(all_scores[1]):.2%} '
      f'majmin: {calc_mean(all_scores[2]):.2%} triads: {calc_mean(all_scores[3]):.2%} '
      f'sevenths: {calc_mean(all_scores[4]):.2%} tetrads: {calc_mean(all_scores[5]):.2%} '
      f'mirex: {calc_mean(all_scores[6]):.2%}')
