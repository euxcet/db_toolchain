import numpy as np

from core.window import Window
from core.imu_data import IMUDataGroup

class IMUWindow():
  def __init__(self, imu_window_length=40, moving_window_length=5):
    self.imu_window_length = imu_window_length
    self.moving_window_length = moving_window_length
    self.imu_window = Window[IMUDataGroup](imu_window_length)
    self.moving_window = Window[int](moving_window_length)

    self.still_state_window = Window[bool](imu_window_length)
    self.direction_state_window = Window[int](imu_window_length)
    self.surface_state_window = Window[int](imu_window_length)
    self.timestamp = 0
  
  def full(self):
    return self.imu_window.full()

  def is_moving(self):
    return self.moving_window.sum() == self.moving_window_length
  
  def push(self, data:IMUDataGroup):
    self.timestamp += 0.005
    self.imu_window.push(data)
    self.moving_window.push(0 if data[0].gyr_norm < 0.05 else 1)
    self.maintain_state()

  def maintain_state(self):
    data = self.imu_window.last()
    if not self.imu_window.full() or not self.still_state_window.full():
      self.still_state_window.push(False)
      self.direction_state_window.push(-1)
      self.surface_state_window.push(-1)
    else:
      self.still_state_window.push(data[1].acc_norm > 9.3 and data[1].acc_norm < 10.5)
      if self.still_state_window.tail(10).count() != 10:
        self.direction_state_window.push(-1)
        self.surface_state_window.push(self.surface_state_window.last())
      else:
        direction = data[1].direction()
        self.direction_state_window.push(direction)
        self.surface_state_window.push(
          direction if self.direction_state_window.tail(15).count(lambda x: x == direction) >= 2 
          else self.surface_state_window.last()
        )