import multiprocessing
import os
import warnings
import yaml
from concurrent.futures import ProcessPoolExecutor
from statistics import mode

import librosa
import mir_eval
import numpy as np
import pandas as pd
from tqdm import tqdm


warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

wav_list = os.listdir('02_audiofiles')
wav_list.sort()

with open('01_utils/feature_config.yaml', 'r') as conf:
    config = yaml.safe_load(conf)

sr = config['sampling_rate']
freq_bins = config['freq_bins']
time_bins = config['time_bins']
hop_length = config['hop_length']
bins_per_octave = config['bins_per_octave']


notes = ['N', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
triads = ['N', 'maj', 'min', 'dim', 'aug', 'sus2', 'sus4']
qualities = ['N', 'maj', 'min', 'dim', 'aug', 'sus2', 'sus4', 'maj6', 'min6',
             '7', 'maj7', 'min7', 'minmaj7', 'dim7', 'hdim7']

sharp = ['B', 'C#', 'D#', 'E', 'F#', 'G#', 'A#']
flat = ['Cb', 'Db', 'Eb', 'Fb', 'Gb', 'Ab', 'Bb']

note_indices = list(range(13))
daug_indices = [note_indices[8:]+note_indices[1:8],
                note_indices[9:]+note_indices[1:9],
                note_indices[10:]+note_indices[1:10],
                note_indices[11:]+note_indices[1:11],
                note_indices[12:]+note_indices[1:12],
                note_indices[2:]+note_indices[1:2],
                note_indices[3:]+note_indices[1:3],
                note_indices[4:]+note_indices[1:4],
                note_indices[5:]+note_indices[1:5],
                note_indices[6:]+note_indices[1:6],
                note_indices[7:]+note_indices[1:7]]
daug_indices = [[0]+x for x in daug_indices]


def split_chord(chord):
    chord_parts = mir_eval.chord.split(chord, reduce_extended_chords=True)

    if chord_parts[0] in flat:
        chord_parts[0] = sharp[flat.index(chord_parts[0])]
    root = notes.index(chord_parts[0])

    if chord_parts[1] in ['maj', 'maj6', '7', 'maj7']:
        triad = 1
    elif chord_parts[1] in ['min', 'min6', 'min7', 'minmaj7']:
        triad = 2
    elif chord_parts[1] in ['dim', 'dim7', 'hdim7']:
        triad = 3
    elif chord_parts[1] in ['aug', 'sus2', 'sus4']:
        triad = triads.index(chord_parts[1])
    else:
        triad = 0

    if chord_parts[1] in qualities:
        quality = qualities.index(chord_parts[1])
    else:
        quality = 0

    if chord_parts[1] in ['maj7', 'minmaj7']:
        tone7 = 3
    elif chord_parts[1] in ['min7', 'hdim7']:
        tone7 = 2
    elif chord_parts[1] == 'dim7':
        tone7 = 1
    else:
        tone7 = 0

    if 'b2' in chord_parts[2] or 'b9' in chord_parts[2]:
        tone9 = 1
    elif '2' in chord_parts[2] or '9' in chord_parts[2]:
        tone9 = 2
    elif 'b3' in chord_parts[2] or '#9' in chord_parts[2]:
        tone9 = 3
    else:
        tone9 = 0

    if '4' in chord_parts[2] or '11' in chord_parts[2]:
        tone11 = 1
    elif 'b5' in chord_parts[2] or '#11' in chord_parts[2]:
        tone11 = 2
    else:
        tone11 = 0

    if '#5' in chord_parts[2] or 'b13' in chord_parts[2]:
        tone13 = 1
    elif '6' in chord_parts[2] or '13' in chord_parts[2]:
        tone13 = 2
    else:
        tone13 = 0

    return [root, triad, quality, tone7, tone9, tone11, tone13]


def classify_c170(root, quality):
    if root == 0:
        c170 = 169
    elif quality == 0:
        c170 = 168
    else:
        c170 = (root - 1) + ((quality - 1) * 12)

    return c170


def classify_c181(root, quality):
    if root == 0:
        c181 = 180
    else:
        c181 = (root - 1) + (quality * 12)

    return c181


def convert_lab2smp(lab_path):
    lab_csv = pd.read_csv(lab_path, header=None, sep='[ \t]', engine='python', names=('End', 'Chord'))
    lab_csv['End_smp'] = librosa.time_to_samples(lab_csv['End'])

    smp_unit_seq = np.empty((7, lab_csv.iat[-1, -1]), dtype=np.int32)

    start_smp_point = 0
    for end_smp_point, chord in zip(lab_csv['End_smp'], lab_csv['Chord']):
        smp_duration = end_smp_point - start_smp_point
        tone_list = split_chord(chord)

        smp_unit_seq[:, start_smp_point:end_smp_point] = np.array([[tone_list[i]]*smp_duration for i in range(7)])
        start_smp_point = end_smp_point

    del lab_csv
    return smp_unit_seq


def convert_smp2frame(smp_unit_seq, clip_idx):
    frame_unit_seq = np.empty((9, time_bins), dtype=np.int32)

    for t in range(time_bins):
        start_smp_point = clip_idx*110250 + t*hop_length
        end_smp_point = clip_idx*110250 + t*hop_length + hop_length

        frame_unit_seq[:7, t] = [mode(smp_unit_seq[i, start_smp_point:end_smp_point]) for i in range(7)]

    frame_unit_seq[7, :] = np.array([classify_c170(a, b) for a, b in zip(frame_unit_seq[0], frame_unit_seq[2])])
    frame_unit_seq[8, :] = np.array([classify_c181(a, b) for a, b in zip(frame_unit_seq[0], frame_unit_seq[2])])

    return frame_unit_seq


def compute_cqt(y, pitch):
    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)

    C = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=freq_bins, bins_per_octave=bins_per_octave)
    C = np.delete(C, time_bins, 1)
    C = np.log(np.abs(C) + 1e-6)

    return C


def generate_features(wav_name):
    song_name = wav_name.replace('.wav', '')
    lab_name = f'{song_name}.lab'
    wav_path = f'02_audiofiles/{wav_name}'
    lab_path = f'02_chordfiles/{lab_name}'

    save_dir = f'03_features/{song_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    smp_unit_seq = convert_lab2smp(lab_path)

    song_len = min(librosa.samples_to_time(len(smp_unit_seq[0])), librosa.get_duration(filename=wav_path))
    y, _ = librosa.load(wav_path, sr=sr, duration=song_len)

    half_n = sr*10 // 2

    if len(y) % half_n >= 1000:
        clip_num = len(y)//half_n - 1
    else:
        clip_num = len(y)//half_n - 2

    clip_arr = np.array([y[i*half_n:i*half_n + 221500] for i in range(clip_num)])

    for clip_idx in range(clip_num):
        cqt = compute_cqt(clip_arr[clip_idx], 0)
        frame_unit_seq = convert_smp2frame(smp_unit_seq, clip_idx)

        np.savez(f'{save_dir}/{song_name}_0_{clip_idx}',
                 cqt=cqt,
                 root=frame_unit_seq[0],
                 triad=frame_unit_seq[1],
                 quality=frame_unit_seq[2],
                 tone7=frame_unit_seq[3],
                 tone9=frame_unit_seq[4],
                 tone11=frame_unit_seq[5],
                 tone13=frame_unit_seq[6],
                 c170=frame_unit_seq[7],
                 c181=frame_unit_seq[8])

        for pitch, daug in zip([i for i in range(-5, 7) if i != 0], daug_indices):
            cqt_daug = compute_cqt(clip_arr[clip_idx], pitch)

            frame_roots_daug = [daug[orig_root] for orig_root in frame_unit_seq[0]]
            frame_c170s_daug = [classify_c170(a, b) for a, b in zip(frame_roots_daug, frame_unit_seq[2])]
            frame_c181s_daug = [classify_c181(a, b) for a, b in zip(frame_roots_daug, frame_unit_seq[2])]

            np.savez(f'{save_dir}/{song_name}_{pitch}_{clip_idx}',
                     cqt=cqt_daug,
                     root=frame_roots_daug,
                     triad=frame_unit_seq[1],
                     quality=frame_unit_seq[2],
                     tone7=frame_unit_seq[3],
                     tone9=frame_unit_seq[4],
                     tone11=frame_unit_seq[5],
                     tone13=frame_unit_seq[6],
                     c170=frame_c170s_daug,
                     c181=frame_c181s_daug)

    return None


if __name__ == "__main__":
    spawn = multiprocessing.set_start_method('spawn')

    with ProcessPoolExecutor(max_workers=6, mp_context=spawn) as executor:
        _ = list(tqdm(executor.map(generate_features, wav_list), total=len(wav_list)))
