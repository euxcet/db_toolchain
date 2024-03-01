from __future__ import annotations

import struct
import numpy as np
from pathlib import Path
from db_train.dataset.base_dataset import BaseDataset
from typing_extensions import override

"""
  data: tuple(imu_data, label)
  imu_data: np.ndarray, shape (200, 6)
  label: int
"""
class IMUGestureDatasetNumpy(BaseDataset):
  def __init__(
      self,
      data_path: Path,
      label_path: Path,
  ) -> None:
    super(IMUGestureDatasetNumpy, self).__init__(
      mode='numpy'
    )
    self.data_path = data_path
    self.label_path = label_path

  @override
  def load(self) -> IMUGestureDatasetNumpy:
    self.np_arrays.append(np.load(self.data_path))
    self.np_arrays.append(np.load(self.label_path).astype(np.int64))
    return self

  def export(self) -> None:
    ...
  
  def augment(self) -> None:
    ...

if __name__ == '__main__':
  dataset = IMUGestureDatasetNumpy(Path('../../../local/dataset/0_ring_gesture_dataset/train_x.npy'), Path('../../../local/dataset/0_ring_gesture_dataset/train_y.npy'))
  dataset.load()
  print(dataset[0])