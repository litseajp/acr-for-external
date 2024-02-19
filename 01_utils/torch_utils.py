import random

import mir_eval
import numpy as np
import torch
from torch.utils.data import Dataset


class MusicDataSet(Dataset):
    def __init__(self, paths, label_class, need_path):
        self.paths = paths
        self.label_class = label_class
        self.need_path = need_path

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        instance_path = self.paths[idx]
        data = np.load(f'03_features/{instance_path}')
        feature = data['cqt']
        label = data[self.label_class]

        if self.need_path:
            return feature, label, instance_path
        else:
            return feature, label


class ReduceLR():
    def __init__(self, patience=2, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.reduce_lr = False
        self.best_loss = None
        self.counter = 0

    def __call__(self, val_loss):
        self.reduce_lr = False
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.reduce_lr = True
                self.counter = 0


class EarlyStopping():
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.early_stop = False
        self.best_loss = None
        self.counter = 0

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def evaluate_chordfile(ref_lab, est_lab):
    (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(ref_lab)
    (est_intervals, est_labels) = mir_eval.io.load_labeled_intervals(est_lab)
    est_intervals, est_labels = mir_eval.util.adjust_intervals(
        est_intervals, est_labels, ref_intervals.min(), ref_intervals.max(),
        mir_eval.chord.NO_CHORD, mir_eval.chord.NO_CHORD)
    (intervals, ref_labels, est_labels) = mir_eval.util.merge_labeled_intervals(
        ref_intervals, ref_labels, est_intervals, est_labels)
    durations = mir_eval.util.intervals_to_durations(intervals)

    root_comparisons = mir_eval.chord.root(ref_labels, est_labels)
    root = mir_eval.chord.weighted_accuracy(root_comparisons, durations)

    thirds_comparisons = mir_eval.chord.thirds(ref_labels, est_labels)
    thirds = mir_eval.chord.weighted_accuracy(thirds_comparisons, durations)

    majmin_comparisons = mir_eval.chord.majmin(ref_labels, est_labels)
    majmin = mir_eval.chord.weighted_accuracy(majmin_comparisons, durations)

    triads_comparisons = mir_eval.chord.triads(ref_labels, est_labels)
    triads = mir_eval.chord.weighted_accuracy(triads_comparisons, durations)

    sevenths_comparisons = mir_eval.chord.sevenths(ref_labels, est_labels)
    sevenths = mir_eval.chord.weighted_accuracy(sevenths_comparisons, durations)

    tetrads_comparisons = mir_eval.chord.tetrads(ref_labels, est_labels)
    tetrads = mir_eval.chord.weighted_accuracy(tetrads_comparisons, durations)

    mirex_comparisons = mir_eval.chord.mirex(ref_labels, est_labels)
    mirex = mir_eval.chord.weighted_accuracy(mirex_comparisons, durations)

    return [root, thirds, majmin, triads, sevenths, tetrads, mirex]


def recomposition_c170(num):
    root_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', ' B']
    quality_list = ['maj', 'min', 'dim', 'aug', 'sus2', 'sus4', 'maj6', 'min6',
                    '7', 'maj7', 'min7', 'minmaj7', 'dim7', 'hdim7']

    if num == 168:
        chord = 'X'
    elif num == 169:
        chord = 'N'
    else:
        root = root_list[(num) % 12]
        quality = quality_list[(num)//12]
        chord = root + ':' + quality

    return chord


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def format_time(start_time, end_time):
    exec_time = int(end_time - start_time)
    exec_hour = exec_time // 3600
    exec_min = (exec_time % 3600) // 60
    exec_sec = (exec_time % 3600) % 60

    return f'{exec_hour:02}:{exec_min:02}:{exec_sec:02}'
