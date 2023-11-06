import torch
import numpy as np
from utils.window import Window
from utils.filter import OneEuroFilter
from model.imu_trajectory_model import TrajectoryLSTMModel
from demo.detector import Detector, DetectorEvent
from sensor.imu_data import IMUData
from sensor import Ring, RingEvent, RingEventType, ring_pool
from sensor.glove import Glove

class TrajectoryDetector(Detector):
  CALCULATE_STEP = 4
  TIMESTAMP_STEP = 0.02
  IMU_WINDOW_LEN = 20
  EXECUTE_INTERVAL = 10
  MOVE_THRESHOLD = 0.1
  def __init__(self, device:Ring|Glove, handler=None, arguments:dict=dict()):
    super(TrajectoryDetector, self).__init__(TrajectoryLSTMModel(), device, handler, arguments)
    self.imu_window = Window[IMUData](self.IMU_WINDOW_LEN)
    self.counter.execute_interval = self.EXECUTE_INTERVAL
    self.touch_up()

  def handle_detector_event(self, event:DetectorEvent):
    if event.detector == self.arguments['gesture_detector_name']:
      if event.data == 'touch_down':
        self.touch_down()
      elif event.data in ['touch_up', 'click', 'double_click']:
        self.touch_up()

  def touch_down(self):
    self.touching = True

  def touch_up(self):
    self.touching = False
    self.one_euro_filter = OneEuroFilter(np.zeros(2), 0)
    self.broadcast_event(-self.last_unstable_move)
    self.last_unstable_move = np.zeros(0, 0)

  def handle_ring_event(self, device, event:RingEvent):
    if event.event_type != RingEventType.imu:
      return
    self.imu_window.push(event.data.to_numpy())
    if self.counter.count() and self.imu_window.full() and self.touching:
      input_tensor = torch.tensor(self.imu_window.to_numpy_float().reshape(1, self.IMU_WINDOW_LEN, 6)).to(self.device)
      output = self.model(input_tensor).detach().cpu().numpy().flatten()
      if np.linalg.norm(output) < self.MOVE_THRESHOLD:
        output = np.zeros_like(output)
        self.last_unstable_move = np.zeros(0, 0)
      move = self.one_euro_filter(output, dt=self.TIMESTAMP_STEP)
      self.last_unstable_move += move
      self.broadcast_event(move)