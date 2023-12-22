import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))

import argparse

import time
from io import BufferedReader
import random
import struct
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from matplotlib import pyplot as plt
from sklearn import metrics
from model.imu_trajectory_model import TrajectoryLSTMModel
# from model import BaselineModel0, LSTMModel


def load_imu_frame(f:BufferedReader):
  try:
    return [struct.unpack('f', f.read(4))[0] for i in range(6)]
  except:
    return None

def load_trajectory_frame(f:BufferedReader):
  try:
    return [struct.unpack('f', f.read(4))[0] for i in range(6)]
  except:
    return None

def extract_record(f_ring:BufferedReader, f_board:BufferedReader):
  imu_window = []
  trajectory_window = []

  while True:
    imu = load_imu_frame(f_ring)
    if imu is None: break
    imu_window.append(imu)

  while True:
    trajectory = load_trajectory_frame(f_board)
    if trajectory is None: break
    trajectory_window.append(trajectory)

  l_i = len(imu_window)
  l_t = len(trajectory_window)

  if l_i == 0 or not (l_i >= l_t * 3.8 and l_i <= l_t * 4.2):
    print('Skip', f_ring, l_i, l_t, l_i / l_t)
    return None, None

  ratio = l_i / l_t
    
  skip_imu = 20
  skip_trajectory = skip_imu // 4
  step = 20
  skip_tail_imu = 80
  imu_input_len = 20
  trajectory_input_len = imu_input_len // 4

  imu_window = imu_window[skip_imu:]
  trajectory_window = np.array(trajectory_window[skip_trajectory:])

  x_imus = []
  ys = []

  for i in range(0, len(imu_window) - skip_tail_imu - imu_input_len, step):
    x_imu = np.array(imu_window[i: i + imu_input_len])
    t_len = 2
    p0 = trajectory_window[int(round(i / ratio)) + trajectory_input_len - t_len]
    p1 = trajectory_window[int(round(i / ratio)) + trajectory_input_len]
    y = np.array([(p1[0] - p0[0]) * 100 / t_len, (p1[1] - p0[1]) * 100 / t_len]).flatten()
    x_imus.append(x_imu)
    ys.append(y)
  return np.array(x_imus)[np.newaxis, :], np.array(ys)[np.newaxis, :]

def get_dataset(datasets):
  x_imus = []
  ys = []
  for dataset in datasets:
    for user in os.listdir(dataset):
      if not user.startswith('.'):
        user_path = os.path.join(dataset, user)
        for folder in os.listdir(user_path):
          if not folder.startswith('.'):
            folder_path = os.path.join(user_path, folder)
            for f in os.listdir(folder_path):
              if f.endswith('.ring'):
                x_i, y = extract_record(open(os.path.join(folder_path, f), 'rb'), open(os.path.join(folder_path, f[:-4] + 'board'), 'rb'))
                if x_i is not None and len(x_i.shape) == 4:
                  x_imus.extend(x_i)
                  ys.extend(y)
  return TensorDataset(torch.tensor(np.concatenate(x_imus).astype('float32')),
                       torch.tensor(np.concatenate(ys).astype('float32')))

class Loss(nn.Module):
  def __init__(self):
    super().__init__()
    self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-8)

  def forward(self, x, y):
    length_loss = torch.mean(torch.abs((torch.norm(x, dim=1) - torch.norm(y, dim=1))))
    angle_loss = 1 - torch.mean(self.cos_sim(x, y))
    return length_loss + angle_loss * 2

def train():
    # reset random seed
    np.random.seed(0)

# train_dataset = get_dataset(['ringboard_dataset', 'ringboard_left_dataset', 'ringboard_left_vertical_dataset', 'ringboard_vertical_dataset'])
    train_dataset = get_dataset(['glove_dataset'])
    print('Size of the train dataset:', len(train_dataset))

    device = 'cpu'
    model = TrajectoryLSTMModel().to(device)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size, num_workers=2)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10)
    criterion = Loss()
    best_loss = 1000.0

    print('Start training.')
    t0 = time.time()
    for epoch in range(5000):
      model.train()
      train_loss = 0
      for imu_batch, y_batch in train_loader:
        imu_batch = imu_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        out = model(imu_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imu_batch.size(0)

      train_loss = train_loss / len(train_loader.dataset)

      if epoch % 50 == 0:
        print(epoch, train_loss, best_loss)

      model.eval()

      if train_loss < best_loss:
        best_loss = train_loss
        torch.save({
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state': optimizer.state_dict(),
        }, 'checkpoint/' + str(epoch) + '_checkpoint.tar')
        torch.save(model.state_dict(), 'trajectory_glove.pth')
      scheduler.step(train_loss)
    print(time.time() - t0)

    return model

if __name__ == '__main__':
  try:
    os.makedirs('checkpoint')
  except:
    pass
  train()
