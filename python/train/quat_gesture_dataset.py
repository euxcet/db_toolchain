import numpy as np
import os.path as osp
from torch.utils.data import Dataset
from dataset.file_dataset import FileDataset

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

def get_quat_gesture_dataset(root:str, class_map=None):
  data, labels = [], []
  file_dataset = FileDataset(root)
  for user, class_, number, _ in file_dataset.records:
    data_filename = osp.join(root, user, class_, number + '_glove.npy')
    new_data = np.load(data_filename).tolist()
    data.extend(new_data)
    labels.extend([map_class(file_dataset.label_id[class_], class_map)] * len(new_data))
  return QuatGestureDataset(np.array(data, dtype=np.float32), np.array(labels))
