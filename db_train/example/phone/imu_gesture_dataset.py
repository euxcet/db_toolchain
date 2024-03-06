from __future__ import annotations

import struct
import numpy as np
from pathlib import Path
from db_train.dataset.hierarchical_dataset import HierarchicalDataset
from db_train.augment.augmentor import Augmentor
from db_train.data.imu_data import IMUData
from db_train.utils.data_utils import slice_data
from db_train.utils.window import Window
from typing_extensions import override
from matplotlib.pylab import plt

"""
  data: tuple(imu_data, label)
  imu_data: np.ndarray, shape (200, 6)
  label: int
"""
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

  def load_imu_data(self, filename: Path) -> np.ndarray:
    acc = []
    gyr = []
    with open(filename, 'rb') as f:
      while data := f.read(24):
        sensor, x, y, z, timestamp = struct.unpack('>ifffq', data)
        if sensor == 1:
          acc.append([x, y, z, timestamp])
        elif sensor == 4:
          gyr.append([x, y, z, timestamp])
    imu = Window(375)
    for i in range(min(len(acc), len(gyr))):
      imu.push(IMUData(acc[i][0], acc[i][1], acc[i][2], gyr[i][0], gyr[i][1], gyr[i][2], 1.0 * gyr[i][3]))
    return imu.pad().to_numpy_float()

  def plot(self, title: str, data) -> None:
    t = np.array([i for i in range(data.shape[0])])
    print(data[:, 0].shape)
    plt.title(title)
    plt.plot(t, data[:, 0])
    plt.plot(t, data[:, 1])
    plt.plot(t, data[:, 2])
    plt.show()

  @override
  def load(self) -> IMUGestureDataset:
    self.file_loader.load()
    for hier in sorted(self.file_loader.hierarchies):
      if (imu_file := hier.get_file('imu')) != None:
        imu = self.load_imu_data(imu_file)
        user, category, round_id = hier.get_hierarchy(0), hier.get_hierarchy(1), hier.get_hierarchy(2)
        # if category == 'point' and user == 'zcc0':
        #   print(imu.shape, hier)
        #   self.plot(str(hier), imu)
        self.data.append((imu, self.label_id[self.label_mapping[category]]))
    return self

  def export(self) -> None:
    ...

if __name__ == '__main__':
  dataset = IMUGestureDataset(Path('../../local/dataset/1007_ring_dataset'))
  dataset.load()