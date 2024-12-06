from __future__ import annotations

import math
import numpy as np
from multipledispatch import dispatch

class IMUData():
  @dispatch(float, float, float, float, float, float, float)
  def __init__(
      self,
      acc_x:float,
      acc_y:float,
      acc_z:float,
      gyr_x:float,
      gyr_y:float,
      gyr_z:float,
      timestamp:float
  ) -> None:
    self.acc_x = acc_x
    self.acc_y = acc_y
    self.acc_z = acc_z
    self.gyr_x = gyr_x
    self.gyr_y = gyr_y
    self.gyr_z = gyr_z
    self.timestamp = timestamp
    self.acc_np = np.array([self.acc_x, self.acc_y, self.acc_z])
    self.gyr_np = np.array([self.gyr_x, self.gyr_y, self.gyr_z])
    self.imu_np = np.concatenate((self.acc_np, self.gyr_np), axis=0)
  
  @dispatch(tuple, float)
  def __init__(self, data:tuple, timestamp:float) -> None:
    if len(data) == 6:
      self.__init__(*data, timestamp)
    else:
      raise NotImplementedError

  @dispatch(dict)
  def __init__(self, data:dict) -> None:
    self.__init__(data['acc_x'], data['acc_y'], data['acc_z'],
                  data['gyr_x'], data['gyr_y'], data['gyr_z'], data['timestamp'])
  
  def __getitem__(self, index:int) -> float:
    return [self.acc_x, self.acc_y, self.acc_z, self.gyr_x, self.gyr_y, self.gyr_z][index]

  def __str__(self) -> str:
    return 'Acc [x: {:.2f}  y: {:.2f}  z: {:.2f}]  Gyr[x: {:.2f}  y: {:.2f}  z: {:.2f}]  Timestamp {:.2f}'.format(
      self.acc_x, self.acc_y, self.acc_z, self.gyr_x, self.gyr_y, self.gyr_z, self.timestamp
    )

  @property
  def gyr_norm(self) -> float:
    return math.sqrt(self.gyr_x * self.gyr_x + self.gyr_y * self.gyr_y + self.gyr_z * self.gyr_z)
  
  @property
  def acc_norm(self) -> float:
    return math.sqrt(self.acc_x * self.acc_x + self.acc_y * self.acc_y + self.acc_z * self.acc_z)

  def to_numpy(self) -> np.ndarray:
    return np.array([self.acc_x, self.acc_y, self.acc_z, self.gyr_x, self.gyr_y, self.gyr_z])

  def sub(self, d: IMUData) -> IMUData:
    return IMUData(self.acc_x - d.acc_x, self.acc_y - d.acc_y, self.acc_z - d.acc_z, \
                   self.gyr_x - d.gyr_x, self.gyr_y - d.gyr_y, self.gyr_z - d.gyr_z, 0.0)