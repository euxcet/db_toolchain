from torch.utils.data import Dataset
from abc import ABC, abstractmethod

class BaseDataset(Dataset, ABC):
  def __init__(self) -> None:
    self.data: list = []

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
    return len(self.data)

  def __getitem__(self, index: int):
    return self.data[index]
