import os
import os.path as osp
import numpy as np
import torch
from data_processing import load_data, load_glove_data
from torch.utils.data import Dataset
from file_dataset import FileDataset

class GestureDataset(Dataset):
  def __init__(self, ring_data, labels):
    '''
    glove_data: np.ndarray, shape=(N, 400, 16, 4)
    ring_data: np.ndarray, shape=(N, 400, 6)
    labels: np.ndarray, shape=(N,)
    '''
    ring_data = np.swapaxes(ring_data, 1, 2).reshape(-1, 6, 1, 200)
    self.ring_data = ring_data
    self.labels = labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    return self.ring_data[index], self.labels[index]

  def get_labels(self):
    return self.labels

def map_class(class_:int, class_map:dict):
  return class_ if class_map is None else class_map[class_]

def get_gesture_dataset(root, class_map=None):
  ring_data, labels = [], []

  file_dataset = FileDataset(root)
  print(file_dataset.label_id)
  
  for user, class_, number, _ in file_dataset.records:
    ring_filename = osp.join(root, user, class_, number + '_ring.bin')
    glove_filename = osp.join(root, user, class_, number + '_glove.bin')
    timestamp_filename = osp.join(root, user, class_, number + '_timestamp.txt')
    if osp.exists(ring_filename) and osp.exists(timestamp_filename):
      data = load_data(user, class_, ring_filename, timestamp_filename)
      ring_data.extend(data)
      labels.extend([map_class(file_dataset.label_id[class_], class_map)] * len(data))

    if osp.exists(glove_filename) and osp.exists(timestamp_filename):
      data = load_glove_data(user, class_, glove_filename, timestamp_filename)
      ring_data.extend(data)
      labels.extend([map_class(file_dataset.label_id[class_], class_map)] * len(data))

  ring_data = np.array(ring_data, dtype=np.float32)
  labels = np.array(labels)
  return GestureDataset(ring_data, labels)
