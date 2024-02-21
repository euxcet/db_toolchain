from abc import ABC
import os.path as osp
from pathlib import Path
from .base_dataset import BaseDataset
from .file_loader import FileLoader
from ..utils.file_utils import load_json
from ..augment.augmentor import Augmentor

class HierarchicalDataset(BaseDataset, ABC):

  CONFIG_FILENAME = 'config.json'
  CONFIG_LABEL_FIELDS = 'label'
  CONFIG_LABEL_MAPPING_FIELDS = 'label_mapping'
  
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
    if self.CONFIG_LABEL_FIELDS in self.config:
      self.label = list(self.config.get(self.CONFIG_LABEL_FIELDS))
      self.label_mapping = dict(self.config.get(self.CONFIG_LABEL_MAPPING_FIELDS))
      self.label_id = {key: id for id, key in enumerate(self.label)}
  
  def augment(self) -> None:
    pass

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    return self.data[index]
