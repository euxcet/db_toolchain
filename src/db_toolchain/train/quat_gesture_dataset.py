import math
import random
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
from dataset.file_dataset import FileDataset
from pyquaternion import Quaternion

def rotate(data):
  rotated_data = []
  for t, skeleton in enumerate(data):
    angle = random.random() * math.pi * 2
    if t % 10 != 0:
      continue
    rotated_skeleton = []
    for i in range(16):
      q = Quaternion(axis=[0, 1, 0], angle=angle) * Quaternion(skeleton[i * 4: i * 4 + 4])
      rotated_skeleton.extend(q.elements.tolist())
    rotated_data.append(rotated_skeleton)
  return rotated_data

class QuatGestureDataset(Dataset):
  def __init__(self, data, labels):
    self.data = data
    self.labels = labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, index):
    return self.data[index], self.labels[index]
    
  def get_labels(self):
    return self.labels

def map_class(class_:int, class_map:dict):
  return class_ if class_map is None else class_map[class_]

def get_quat_gesture_dataset(roots:list[str], class_map=None):
  for root in roots:
    data, labels = [], []
    file_dataset = FileDataset(root)
    for user, class_, number, _ in file_dataset.records:
      print(user, class_, number)
      data_filename = osp.join(root, user, class_, number + '_glove.npy')
      new_data = np.load(data_filename).tolist()
      new_data = rotate(new_data)
      data.extend(new_data)
      labels.extend([map_class(file_dataset.label_id[class_], class_map)] * len(new_data))
  return QuatGestureDataset(np.array(data, dtype=np.float32), np.array(labels))
