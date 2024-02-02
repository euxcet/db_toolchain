from __future__ import annotations

import numpy as np
from multipledispatch import dispatch
from pyquaternion import Quaternion

class QuaternionData():
  @dispatch(float, float, float, float, float)
  def __init__(
      self,
      w:float,
      x:float,
      y:float,
      z:float,
      timestamp:float
  ) -> None:
    self.quaternion = Quaternion(w, x, y, z)
    self.w = w
    self.x = x
    self.y = y
    self.z = z
    self.timestamp = timestamp

  @dispatch(tuple, float)
  def __init__(self, data:tuple, timestamp:float) -> None:
    if len(data) == 4:
      self.__init__(*data, timestamp)
    else:
      raise NotImplementedError
  
  def to_numpy(self) -> np.ndarray:
    return np.array([self.w, self.x, self.y, self.z])
