import os
import torch
import numpy as np
from utils.window import Window
from utils.filter import OneEuroFilter
from model.imu_trajectory_model import TrajectoryLSTMModel
from demo.detector import Detector, DetectorEvent
from sensor.basic_data import IMUData
from sensor import Ring, RingEvent, RingEventType
from sensor.glove import Glove, GloveEvent, GloveEventType
from sensor.glove_data import GloveIMUJointName

class TrajectoryDetector(Detector):
  def __init__(self, name:str, device:str, devices:dict, gesture_detector_name:str, checkpoint_file:str,
               imu_window_length:int, execute_interval:int, timestamp_step:float, move_threshold:float, handler=None):
    super(TrajectoryDetector, self).__init__(name=name, model=TrajectoryLSTMModel(), device=devices[device],
                                             checkpoint_file=checkpoint_file, handler=handler)
    self.touching = False
    self.gesture_detector_name = gesture_detector_name
    self.imu_window_length = imu_window_length
    self.timestamp_step = timestamp_step
    self.move_threshold = move_threshold
    self.imu_window = Window[IMUData](imu_window_length)
    self.one_euro_filter = OneEuroFilter(np.zeros(2), 0)
    self.stable_window = Window(5)
    self.last_unstable_move = Window(20)
    self.counter.execute_interval = execute_interval

  def touch_down(self):
    self.touching = True

  def touch_up(self):
    self.touching = False
    self.one_euro_filter = OneEuroFilter(np.zeros(2), 0)
    if self.last_unstable_move.capacity() > 0:
      self.broadcast_event(-self.last_unstable_move.sum())
      self.last_unstable_move.clear()

  def detect(self, data):
    self.imu_window.push(data)
    if self.counter.count() and self.imu_window.full() and self.touching:
      input_tensor = torch.tensor(self.imu_window.to_numpy_float().reshape(1, self.imu_window_length, 6)).to(self.device)
      output = self.model(input_tensor).detach().cpu().numpy().flatten()
      self.stable_window.push(np.linalg.norm(output) < self.move_threshold)
      if self.stable_window.last():
        output = np.zeros_like(output)
      if self.stable_window.full() and self.stable_window.all():
        self.last_unstable_move.clear()
      move = self.one_euro_filter(output, dt=self.timestamp_step)
      self.last_unstable_move.push(move)
      self.broadcast_event(move)

  def handle_detector_event(self, event:DetectorEvent):
    if event.detector == self.gesture_detector_name:
      if event.data == 'touch_down':
        self.touch_down()
      elif event.data in ['touch_up', 'click', 'double_click']:
        self.touch_up()

  def handle_ring_event(self, device, event:RingEvent):
    if event.event_type == RingEventType.imu:
      self.detect(event.data.to_numpy())

  def handle_glove_event(self, device, event:GloveEvent):
    if event.event_type == GloveEventType.pose:
      self.detect(event.data.get_imu_data(GloveIMUJointName.INDEX_INTERMEDIATE).to_numpy())