import torch
import struct
import numpy as np
import os.path as osp
from torch.utils.data import TensorDataset
from dataset.file_dataset import FileDataset

def extract_record(ring_filename:str, board_filename:str):
  imu_window = []
  trajectory_window = []

  with open(ring_filename, 'rb') as file:
    while True:
      data = file.read(24)
      if not data: break
      imu_window.append(struct.unpack('6f', data))

  with open(board_filename, 'rb') as file:
    while True:
      data = file.read(24)
      if not data: break
      trajectory_window.append(struct.unpack('6f', data))

  l_i = len(imu_window)
  l_t = len(trajectory_window)

  if l_i == 0 or not (l_i >= l_t * 3.8 and l_i <= l_t * 4.2):
    print(f'Ignore the record[{ring_filename}] due to unstable frame rate.')
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

def get_trajectory_dataset(roots:list[str]):
  xs = []
  ys = []
  for root in roots:
    file_dataset = FileDataset(root, has_label=False)
    for user, class_, _, _, files in file_dataset.records:
      if len(files) == 2:
        if 'ring' in files[0]:
          x, y = extract_record(osp.join(root, user, class_, files[0]), osp.join(root, user, class_, files[1]))
        else:
          x, y = extract_record(osp.join(root, user, class_, files[1]), osp.join(root, user, class_, files[0]))
        if x is not None:
          xs.extend(x)
          ys.extend(y)
  return TensorDataset(torch.tensor(np.concatenate(xs).astype('float32')),
                       torch.tensor(np.concatenate(ys).astype('float32')))