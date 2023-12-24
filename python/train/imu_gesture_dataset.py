import struct
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
from dataset.file_dataset import FileDataset
from sensor.basic_data import IMUData
from utils.data_utils import slice_data

class IMUGestureDataset(Dataset):
  def __init__(self, ring_data, labels):
    self.ring_data = np.swapaxes(ring_data, 1, 2).reshape(-1, 6, 1, 200)
    self.labels = labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    return self.ring_data[index], self.labels[index]
    
  def get_labels(self):
    return self.labels

def load_glove_data(filename:str):
  glove_data = []
  with open(filename, 'rb') as f:
    data = f.read(320)
    while len(data) > 0:
      index = struct.unpack('i', data[:4])[0]
      gyr_x, gyr_y, gyr_z, acc_x, acc_y, acc_z = struct.unpack('<ffffff', data[88:112])
      timestamp = struct.unpack('d', data[312:])[0]
      glove_data.append(IMUData(acc_x * -9.8, acc_y * -9.8, acc_z * -9.8, gyr_x, gyr_y, gyr_z, timestamp))
      data = f.read(320)
  return glove_data

def load_ring_data(filename:str):
  ring_data = []
  with open(filename, 'rb') as f:
    data = f.read(36)
    while len(data) > 0:
      index, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, timestamp = struct.unpack('<iffffffd', data[:36])
      ring_data.append(IMUData(acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, timestamp))
      data = f.read(36)
  return ring_data

def load_timestamp_data(filename:str):
  with open(filename, 'r') as f:
    lines = list(filter(lambda x: len(x) > 0, [x.strip() for x in f.readlines()]))
    return [list(map(float, x.strip().split(' '))) for x in lines]

def map_class(class_:int, class_map:dict):
  return class_ if class_map is None else class_map[class_]

def get_imu_gesture_dataset(roots:list[str], class_map=None):
  data, labels = [], []
  for root in roots:
    file_dataset = FileDataset(root)
    for user, class_, number, _ in file_dataset.records:
      ring_filename = osp.join(root, user, class_, number + '_ring.bin')
      glove_filename = osp.join(root, user, class_, number + '_glove.bin')
      timestamp_filename = osp.join(root, user, class_, number + '_timestamp.txt')
      if osp.exists(ring_filename) and osp.exists(timestamp_filename):
        sliced_data = slice_data(load_ring_data(ring_filename), load_timestamp_data(timestamp_filename))
        data.extend(sliced_data)
        labels.extend([map_class(file_dataset.label_id[class_], class_map)] * len(sliced_data))
      if osp.exists(glove_filename) and osp.exists(timestamp_filename):
        sliced_data = slice_data(load_glove_data(glove_filename), load_timestamp_data(timestamp_filename))
        data.extend(sliced_data)
        labels.extend([map_class(file_dataset.label_id[class_], class_map)] * len(sliced_data))
  return IMUGestureDataset(np.array(data, dtype=np.float32), np.array(labels))
