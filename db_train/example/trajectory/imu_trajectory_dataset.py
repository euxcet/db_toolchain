from __future__ import annotations

import torch
import struct
import numpy as np
from pathlib import Path
from db_train.dataset.hierarchical_dataset import HierarchicalDataset
from db_train.augment.augmentor import Augmentor
from db_train.data.imu_data import IMUData
from db_train.utils.data_utils import slice_data
from typing_extensions import override

class IMUTrajectoryDataset(HierarchicalDataset):
  def __init__(
      self,
      root: Path,
      augmentor: Augmentor = None,
  ) -> None:
    super(IMUTrajectoryDataset, self).__init__(
      root=root,
      augmentor=augmentor,
      mode='tensor',
    )

  def extract_record(self, ring_filename: Path, board_filename: Path):
    ring_data = []
    with open(ring_filename, 'rb') as file:
      while data := file.read(24):
        ring_data.append(struct.unpack('6f', data))

    board_data = []
    with open(board_filename, 'rb') as file:
      while data := file.read(24):
        board_data.append(struct.unpack('6f', data))

    # The frame rate of the ring data should be approximately 4 times the frame rate of the touchboard data.
    len_ring_data = len(ring_data)
    len_board_data = len(board_data)

    if len_ring_data == 0 or len_ring_data < len_board_data * 3.8 or len_ring_data > len_board_data * 4.2:
      print(f'Ignore the record[{ring_filename}] due to unstable frame rate.')
      return None, None

    ratio = len_ring_data / len_board_data

    # skip the first 20 frames and the last 80 frames
    skip_first_frame = 20
    skip_last_frame = 80
    ring_data = ring_data[skip_first_frame : -skip_last_frame]
    board_data = np.array(board_data[skip_first_frame // 4 : -skip_last_frame // 4])
    
    # The model is fitted to the distance(velocity) swiped on the touchpad when the input IMU data segment ends.
    # x -> model -> y
    xs = []
    ys = []
    # sampling interval
    step = 20
    # length of the IMU data input to the model
    imu_input_len = 20
    board_input_len = imu_input_len // 4
    # interval between the starting point and the ending point of the distance
    pos_interval = 2
    scale = 100

    for i in range(0, len(ring_data) - imu_input_len, step):
      if int(round(i / ratio)) + board_input_len < len(board_data):
        x = np.array(ring_data[i: i + imu_input_len])
        start_pos = board_data[int(round(i / ratio)) + board_input_len - pos_interval]
        end_pos = board_data[int(round(i / ratio)) + board_input_len]
        y = np.array([
          (end_pos[0] - start_pos[0]) * scale / pos_interval,
          (end_pos[1] - start_pos[1]) * scale / pos_interval,
        ]).flatten()
        xs.append(x)
        ys.append(y)

    return np.array(xs)[np.newaxis, :], np.array(ys)[np.newaxis, :]

  @override
  def load(self) -> IMUTrajectoryDataset:
    self.file_loader.load()
    xs = []
    ys = []
    for hier in sorted(self.file_loader.hierarchies):
      if (ring_file := hier.get_file('ring')) != None and \
         (board_file := hier.get_file('board')) != None:
        x, y = self.extract_record(ring_file, board_file)
        if x is not None:
          xs.extend(x)
          ys.extend(y)
    self.tensors = [torch.tensor(np.concatenate(xs).astype('float32')),
                    torch.tensor(np.concatenate(ys).astype('float32')),]
    return self

  def export(self) -> None:
    ...

if __name__ == '__main__':
  dataset = IMUTrajectoryDataset(Path('../../../local/dataset/1113_trajectory_dataset/1113_right_dataset'))
  dataset.load()