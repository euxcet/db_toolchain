from pathlib import Path
from db_train.dataset.hierarchical_dataset import HierarchicalDataset
from db_train.augment.augmentor import Augmentor

class IMUGestureDataset(HierarchicalDataset):
  def __init__(
      self,
      root: Path,
      augmentor: Augmentor = None,
  ) -> None:
    super(IMUGestureDataset, self).__init__(
      root=root,
      augmentor=augmentor,
    )

  def load(self) -> None:
    self.file_loader.load()

  def export(self) -> None:
    ...
