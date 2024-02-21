from abc import ABC
import os.path as osp
from pathlib import Path
from .base_dataset import BaseDataset
from .file_loader import FileLoader
from ..utils.file_utils import load_json
from ..augment.augmentor import Augmentor

class HierarchicalDataset(BaseDataset, ABC):

  CONFIG_FILENAME = 'config.json'
  CONFIG_LABEL_FIELDS = ['label', 'labels']
  
  def __init__(
      self,
      root: Path,
      augmentor: Augmentor = None,
  ) -> None:
    super(HierarchicalDataset, self).__init__()
    self.root = root
    self.file_loader = FileLoader(self.root)
    self.augmentor = augmentor
    self.config = dict(load_json(osp.join(self.root, self.CONFIG_FILENAME)))
    for label_field in self.CONFIG_LABEL_FIELDS:
      if label_field in self.config:
        self.labels = self.config.get(label_field)
  
  def augment(self) -> None:
    pass

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    return self.data[index]
