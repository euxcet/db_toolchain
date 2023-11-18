from __future__ import annotations

import math
import numpy as np
from multipledispatch import dispatch

class IMUData():
  @dispatch(float, float, float, float, float, float, float)
  def __init__(self, acc_x:float, acc_y:float, acc_z:float,
               gyr_x:float, gyr_y:float, gyr_z:float, timestamp:float):
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

  @dispatch(dict)
  def __init__(self, data:dict):
    self.__init__(data['acc_x'], data['acc_y'], data['acc_z'],
                  data['gyr_x'], data['gyr_y'], data['gyr_z'], data['timestamp'])
  
  def __getitem__(self, index):
    return [self.acc_x, self.acc_y, self.acc_z, self.gyr_x, self.gyr_y, self.gyr_z][index]

  def __str__(self):
    return 'Acc [x: {:.2f}  y: {:.2f}  z: {:.2f}]  Gyr[x: {:.2f}  y: {:.2f}  z: {:.2f}]  Timestamp {:.2f}'.format(
      self.acc_x, self.acc_y, self.acc_z, self.gyr_x, self.gyr_y, self.gyr_z, self.timestamp
    )

  @property
  def gyr_norm(self):
    return math.sqrt(self.gyr_x * self.gyr_x + self.gyr_y * self.gyr_y + self.gyr_z * self.gyr_z)
  
  @property
  def acc_norm(self):
    return math.sqrt(self.acc_x * self.acc_x + self.acc_y * self.acc_y + self.acc_z * self.acc_z)

  def to_numpy(self):
    return np.array([self.acc_x, self.acc_y, self.acc_z, self.gyr_x, self.gyr_y, self.gyr_z])

class IMUDataGroup():
  def __init__(self, data:list[IMUData]):
    self.data = data

  def __getitem__(self, index) -> IMUData:
    return self.data[index]

  def __str__(self):
    return '\n'.join(map(str, self.data)) + '\n'

  def to_numpy(self):
    return np.array([x[j] for x in self.data for j in range(6)])

  @property
  def timestamp(self):
    return self.data[0].timestamp