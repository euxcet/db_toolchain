import struct
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from .data_processing import load_data, load_glove_data
from .file_dataset import FileDataset
from sensor.basic_data import IMUData
from utils.window import Window

class GestureDataset(Dataset):
  def __init__(self, ring_data, labels):
    ring_data = np.swapaxes(ring_data, 1, 2).reshape(-1, 6, 1, 200)
    self.ring_data = ring_data
    self.labels = labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    return self.ring_data[index], self.labels[index]
    
  def get_labels(self):
    return self.labels

def load_glove_data_(filename):
    glove_data = []
    with open(filename, "rb") as f:
      data = f.read(320)
      while len(data) > 0:
        index = struct.unpack("i", data[:4])[0]
        gyr_x, gyr_y, gyr_z, acc_x, acc_y, acc_z = struct.unpack('<ffffff', data[88:112])
        timestamp = struct.unpack("d", data[312:])[0]
        glove_data.append(IMUData(acc_x * -9.8, acc_y * -9.8, acc_z * -9.8, gyr_x, gyr_y, gyr_z, timestamp))
        data = f.read(320)
    return glove_data

def load_ring_data(filename):
  ring_data = []
  with open(filename, "rb") as f:
    data = f.read(36)
    while len(data) > 0:
      index, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, timestamp = struct.unpack('<iffffffd', data[:36])
      ring_data.append(IMUData(acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, timestamp))
      data = f.read(36)
  return ring_data

def load_timestamp_data(filename):
  timestamp_data = []
  with open(filename, "r") as f:
    data = f.readline()
    while len(data) > 0:
      timestamp_data.append(list(map(float, data.strip().split(' '))))
      data = f.readline()
  return timestamp_data

def load_glove_data(user, action, glove_filename, timestamp_filename, plot_data=False):
    raw_ring_data = load_glove_data_(glove_filename)
    raw_timestamp_data = load_timestamp_data(timestamp_filename)

    ring_data = []
    ring_pointer = 0

    for timestamp in raw_timestamp_data:
        start_timestamp, end_timestamp = timestamp[0], timestamp[1]
        while ring_pointer < len(raw_ring_data) and raw_ring_data[ring_pointer][1] < start_timestamp:
            ring_pointer += 1
        ring_data_single_action = []
        while ring_pointer < len(raw_ring_data) and raw_ring_data[ring_pointer][1] < end_timestamp:
            ring_data_single_action.append(raw_ring_data[ring_pointer][0])
            ring_pointer += 1
        ring_data.append(ring_data_single_action)
    
    # cut frames to 200 * 2
    empty_data_to_pop = []
    for i in range(len(ring_data)):
        if len(ring_data[i]) > 200:
            ring_data[i] = ring_data[i][:200]
        elif len(ring_data[i]) > 180:
            ring_data[i] += [ring_data[i][-1]] * (200 - len(ring_data[i])) # repeat the last frame, shallow copy
        else:
            empty_data_to_pop.append(i)
    for i in empty_data_to_pop[::-1]:
        # pop error data
        ring_data.pop(i)


    ring_data = [np.array(ring_data[i]) for i in range(len(ring_data))]
    return ring_data

def pad_or_cut(data, length:int):
  if len(data) < length:
    return data + data[-1:] * (length - len(data))
  else:
    return data[:length]

def load_data(ring_filename, timestamp_filename, fps=200, lowest_fps=180, highest_fps=220):
  print(ring_filename)
  raw_ring_data:list[IMUData] = load_ring_data(ring_filename)
  raw_timestamp_data = load_timestamp_data(timestamp_filename)

  data:list[Window] = [Window(fps) for _ in range(len(raw_timestamp_data))]
  raw_data_pointer = 0
  for interval, sample_data in zip(raw_timestamp_data, data):
    while raw_data_pointer < len(raw_ring_data) and raw_ring_data[raw_data_pointer].timestamp <= interval[1]:
      if raw_ring_data[raw_data_pointer].timestamp >= interval[0]:
        sample_data.push(raw_ring_data[raw_data_pointer])
      raw_data_pointer += 1

  data = list(filter(lambda x:x.capacity() >= lowest_fps and x.push_count <= highest_fps, data))
  data = [d.pad().to_numpy_float() for d in data]
  return data


def map_class(class_:int, class_map:dict):
    return class_ if class_map is None else class_map[class_]

def get_gesture_dataset(root, class_map=None):
    ring_data, labels = [], []

    file_dataset = FileDataset(root)
    
    for user, class_, number, files in file_dataset.records:
      print(files)
      ring_filename = osp.join(root, user, class_, number + '_ring.bin')
      glove_filename = osp.join(root, user, class_, number + '_glove.bin')
      timestamp_filename = osp.join(root, user, class_, number + '_timestamp.txt')
      if osp.exists(ring_filename) and osp.exists(timestamp_filename):
        data = load_data(ring_filename, timestamp_filename)
        ring_data.extend(data)
        labels.extend([map_class(file_dataset.label_id[class_], class_map)] * len(data))

      if osp.exists(glove_filename) and osp.exists(timestamp_filename):
        data = load_glove_data(glove_filename, timestamp_filename)
        ring_data.extend(data)
        labels.extend([map_class(file_dataset.label_id[class_], class_map)] * len(data))

    ring_data = np.array(ring_data, dtype=np.float32)
    labels = np.array(labels)
    return GestureDataset(ring_data, labels)
