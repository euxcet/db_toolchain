import numpy as np
from torch.utils.data import Dataset
from abc import ABC, abstractmethod

class BaseDataset(Dataset, ABC):

  MODE_LIST: list[str] = ['list', 'tensor', 'numpy']

  def __init__(self, mode: str = 'list') -> None:
    assert mode in self.MODE_LIST
    self.mode = mode
    self.data: list = []
    self.tensors: list = []
    self.np_arrays: list = [] # index 0: imu_data, index 1: label

  @abstractmethod
  def load(self) -> None:
    ...

  @abstractmethod
  def export(self) -> None:
    ...

  @abstractmethod
  def augment(self) -> None:
    ...

  def __len__(self):
    if self.mode == 'list':
      return len(self.data)
    elif self.mode == 'tensor':
      return self.tensors[0].size(0)
    elif self.mode == 'numpy':
      return self.np_arrays[0].shape[0]

  def __getitem__(self, index: int):
    if self.mode == 'list':
      return self.data[index]
    elif self.mode == 'tensor':
      return tuple(tensor[index] for tensor in self.tensors)
    elif self.mode == 'numpy':
      return tuple(np_array[index] for np_array in self.np_arrays)
