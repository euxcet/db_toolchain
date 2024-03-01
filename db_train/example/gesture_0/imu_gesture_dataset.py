from __future__ import annotations

import struct
from pathlib import Path
from db_train.dataset.base_dataset import BaseDataset
from db_train.augment.augmentor import Augmentor
from db_train.data.imu_data import IMUData
from db_train.utils.data_utils import slice_data
from typing_extensions import override

"""
  data: tuple(imu_data, label)
  imu_data: np.ndarray, shape (200, 6)
  label: int
"""
class IMUGestureDataset(BaseDataset):
  def __init__(
      self,
      augmentor: Augmentor = None,
  ) -> None:
    super(IMUGestureDataset, self).__init__(
      augmentor=augmentor,
    )

  def load_ring_data(self, filename: Path):
    ring_data = []
    with open(filename, 'rb') as f:
      while len(data := f.read(36)) > 0:
        index, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, timestamp = struct.unpack('<iffffffd', data)
        ring_data.append(IMUData(acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, timestamp))
    return ring_data

  @override
  def load(self) -> IMUGestureDataset:
    for hier in sorted(self.file_loader.hierarchies):
      if (ring_file := hier.get_file('ring')) != None and \
         (timestamp_file := hier.get_file('timestamp')) != None:
        user, category, round_id = hier.get_hierarchy(0), hier.get_hierarchy(1), hier.get_hierarchy(2)
        self.data.extend(map(
          lambda x: (x, self.label_id[self.label_mapping[category]]),
          slice_data(self.load_ring_data(ring_file), self.load_timestamp_data(timestamp_file))
        ))
    return self

  def export(self) -> None:
    ...

if __name__ == '__main__':
  dataset = IMUGestureDataset(Path('../../local/dataset/0_ring_gesture_dataset'))
  dataset.load()