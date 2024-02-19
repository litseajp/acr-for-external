import argparse
import datetime
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from 01_utils.torch_utils import MusicDataSet, ReduceLR, EarlyStopping, fix_seed, seed_worker, format_time
from 04_BTCmodel.model import BTC_model
from 04_CRNNmodel.model import CRNN_model


warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['CRNN', 'BTC'], default='CRNN')
parser.add_argument('--index', type=int, default=1)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

if torch.cuda.is_available():
    print('GPU Check: OK')
    device = 'cuda'
else:
    print('GPU Check: NG')
    sys.exit()

fix_seed(args.seed)

with open(f'04_{args.model}model/model_config.yaml', 'r') as conf:
    config = yaml.safe_load(conf)

if args.model == 'CRNN':
    model = CRNN_model(config=config['model']).to(device)
elif args.model == 'BTC':
    model = BTC_model(config=config['model']).to(device)
else:
    raise NotImplementedError


save_dir = f'04_{args.model}model/{datetime.date.today()}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

batch_size = config['experiment']['batch_size']
max_epochs = config['experiment']['max_epochs']

learning_rate = config['experiment']['learning_rate']
reduce_factor = config['experiment']['reduce_factor']
min_lr = config['experiment']['min_lr']

n_chords = config['model']['n_chords']
class_label = f'c{n_chords}'

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

reduce_lr = ReduceLR()
early_stopping = EarlyStopping()


kf = np.load('01_utils/split_indices.npz')

train_feature_paths = kf[f'train{args.index}']
val_feature_paths = kf[f'val{args.index}']

g = torch.Generator()
g.manual_seed(0)

train_dataset = MusicDataSet(train_feature_paths, class_label, False)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True,
                              shuffle=True, worker_init_fn=seed_worker, generator=g)

val_dataset = MusicDataSet(val_feature_paths, class_label, False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True,
                            shuffle=True, worker_init_fn=seed_worker, generator=g)


std_path = f'01_utils/standardization/std_{args.index}.npz'

if not os.path.exists(std_path):
    if not os.path.exists('01_utils/standardization'):
        os.makedirs('01_utils/standardization')

    std_sum = np.float64(0)
    std_sq_sum = np.float64(0)
    std_cnt = np.int64(0)

    for features, _ in tqdm(train_dataloader):
        std_sum += features.sum().item()
        std_sq_sum += features.pow(2).sum().item()
        std_cnt += np.prod(features.shape)

    mean = std_sum / std_cnt
    std = np.sqrt(std_sq_sum/std_cnt - np.square(mean))

    np.savez(std_path, mean=mean, std=std)
    print('std Calculation Completed.')
else:
    std_data = np.load(std_path)
    mean = torch.tensor(std_data['mean'])
    std = torch.tensor(std_data['std'])
    print('std Data Loaded.')


train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []

train_batches_n = len(train_dataloader)
val_batches_n = len(val_dataloader)

start_time = time.time()

print('\n==================== Model Training ====================')
for epoch in range(max_epochs):
    model.train()

    train_total_cnt = 0
    train_correct_cnt = 0
    train_loss_sum = 0

    if epoch > 0:
        print(f'Epoch {epoch:d}: train_loss {train_loss_list[-1]:.4f}, train_acc {train_acc_list[-1]:.2%}, '
              f'val_loss {val_loss_list[-1]:.4f}, val_acc {val_acc_list[-1]:.2%}')

    for train_features, train_labels in tqdm(train_dataloader, leave=False, desc=f'Epoch{epoch+1}'):
        train_features = train_features.to(device)
        train_labels = train_labels.to(device, dtype=torch.long)

        train_features.requires_grad = True
        train_features = train_features.permute(0, 2, 1)
        train_features = (train_features - mean) / std

        optimizer.zero_grad()

        outputs = model(train_features)
        _, prediction = torch.max(outputs, 2)

        train_loss = loss_func(outputs.view(-1, n_chords), train_labels.view(-1))
        train_loss.backward()

        optimizer.step()

        train_total_cnt += train_labels.view(-1).shape[0]
        train_correct_cnt += (prediction == train_labels).sum().item()
        train_loss_sum += train_loss.item()

    with torch.no_grad():
        model.eval()

        val_total_cnt = 0
        val_correct_cnt = 0
        val_loss_sum = 0

        for val_features, val_labels in val_dataloader:
            val_features = val_features.to(device)
            val_labels = val_labels.to(device, dtype=torch.long)

            val_features = val_features.permute(0, 2, 1)
            val_features = (val_features - mean) / std

            outputs = model(val_features)
            _, prediction = torch.max(outputs, 2)

            val_loss = loss_func(outputs.view(-1, n_chords), val_labels.view(-1))

            val_total_cnt += val_labels.view(-1).shape[0]
            val_correct_cnt += (prediction == val_labels).sum().item()
            val_loss_sum += val_loss.item()

    train_loss_list.append(train_loss_sum / train_batches_n)
    train_acc_list.append(train_correct_cnt / train_total_cnt)
    val_loss_list.append(val_loss_sum / val_batches_n)
    val_acc_list.append(val_correct_cnt / val_total_cnt)

    reduce_lr(val_loss_list[-1])
    if reduce_lr.reduce_lr:
        old_lr = optimizer.param_groups[0]['lr']
        new_lr = max(old_lr * reduce_factor, min_lr)
        optimizer.param_groups[0]['lr'] = new_lr

    early_stopping(val_loss_list[-1])
    if early_stopping.early_stop:
        break

    if epoch > 0:
        print('\033[1A', end='')

print(f'Epoch {epoch:d}: train_loss {train_loss_list[-1]:.4f}, train_acc {train_acc_list[-1]:.2%}, '
      f'val_loss {val_loss_list[-1]:.4f}, val_acc {val_acc_list[-1]:.2%}')

torch.save(model.state_dict(), f'{save_dir}/model_{args.index}.pth')

exec_time = format_time(start_time, time.time())

print('Training Finished. Time:' + exec_time)


ep_arr = np.array(list(range(len(train_acc_list))))

fig1, ax1 = plt.subplots()

ax1.plot(ep_arr, train_acc_list, label='Train')
ax1.plot(ep_arr, val_acc_list, label='Val')

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy [%]')
ax1.legend()

plt.savefig(f'{save_dir}/acc_{args.index}.png')

fig2, ax2 = plt.subplots()

ax2.plot(ep_arr, train_loss_list, label='Train')
ax2.plot(ep_arr, val_loss_list, label='Val')

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()

plt.savefig(f'{save_dir}/loss_{args.index}.png')
