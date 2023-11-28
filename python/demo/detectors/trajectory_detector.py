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
from sensor.glove_data import GloveData, GloveIMUJointName

class TrajectoryDetector(Detector):
  TIMESTAMP_STEP = 0.02
  IMU_WINDOW_LEN = 20
  MOVE_THRESHOLD = 0.1
  def __init__(self, device:Ring|Glove=None, handler=None, arguments:dict=dict()):
    super(TrajectoryDetector, self).__init__(
      model=TrajectoryLSTMModel(), device=device,
      handler=handler, arguments=arguments)
    self.imu_window = Window[IMUData](self.IMU_WINDOW_LEN)
    self.counter.execute_interval = self.arguments['execute_interval']
    self.touching = False
    self.one_euro_filter = OneEuroFilter(np.zeros(2), 0)
    self.stable_window = Window(5)
    self.last_unstable_move = Window(20)

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
      input_tensor = torch.tensor(self.imu_window.to_numpy_float().reshape(1, self.IMU_WINDOW_LEN, 6)).to(self.device)
      output = self.model(input_tensor).detach().cpu().numpy().flatten()

      self.stable_window.push(np.linalg.norm(output) < self.MOVE_THRESHOLD)
      if self.stable_window.last():
        output = np.zeros_like(output)
      # if self.stable_windsw.full() and self.stable_window.all():
      #   self.last_unstable_move.clear()
      move = self.one_euro_filter(output, dt=self.TIMESTAMP_STEP)
      self.last_unstable_move.push(move)
      self.broadcast_event(move)

  def handle_detector_event(self, event:DetectorEvent):
    if event.detector == self.arguments['gesture_detector_name']:
      if event.data == 'touch_down':
        self.touch_down()
      elif event.data in ['touch_up', 'click', 'double_click']:
        self.touch_up()

  def handle_ring_event(self, device, event:RingEvent):
    if event.event_type == RingEventType.imu:
      self.detect(event.data.to_numpy())

  def handle_glove_event(self, device, event:GloveEvent):
    if event.event_type == GloveEventType.pose:
      data:GloveData = event.data
      # print(data.get_imu_data(GloveIMUJointName.INDEX_INTERMEDIATE).to_numpy())
      self.detect(data.get_imu_data(GloveIMUJointName.INDEX_INTERMEDIATE).to_numpy())