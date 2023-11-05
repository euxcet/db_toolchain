import torch
import numpy as np
from sensor.imu_data import IMUData
from utils.window import Window
from utils.filter import OneEuroFilter
from model.imu_trajectory_model import TrajectoryLSTMModel
from demo.detector import Detector
from sensor import Ring, RingEvent, RingEventType, ring_pool
from sensor.glove import Glove

class TrajectoryDetector(Detector):
  CALCULATE_STEP = 4
  TIMESTAMP_STEP = 0.02
  IMU_WINDOW_LEN = 20
  EXECUTE_INTERVAL = 10
  MOVE_THRESHOLD = 0.2
  def __init__(self, checkpoint_path, device:Ring|Glove, handler=None):
    super(TrajectoryDetector, self).__init__(TrajectoryLSTMModel(), checkpoint_path)
    self.handler = handler
    self.touching = False
    self.imu_window = Window[IMUData](self.IMU_WINDOW_LEN)
    self.filter = OneEuroFilter(0, np.zeros(2))
    self.counter.execute_interval = self.EXECUTE_INTERVAL
    if type(device) is Ring:
      ring_pool.bind_ring(self.handle_ring_imu_event, ring=device)
    elif type(device) is Glove:
      # TODO
      pass

  def handle_detector_event(self, detector, event):
    # touch event
    pass

  def touch_down(self):
    self.touching = True

  def touch_up(self):
    self.touching = False
    self.filter = OneEuroFilter(0, np.zeros(2))

  def handle_ring_imu_event(self, device, event:RingEvent):
    if event.event_type != RingEventType.imu:
      return
    self.imu_window.push(event.data)
    if self.counter.count() and self.imu_window.full():
      input_tensor = torch.tensor(self.imu_window.to_numpy_float().reshape(1, self.IMU_WINDOW_LEN, 6)).to(self.device)
      output = self.model(input_tensor).detach().cpu().numpy().flatten()
      if np.linalg.norm(output) < self.MOVE_THRESHOLD:
        output = np.zeros_like(output)
      move = self.filter(output, dt=self.TIMESTAMP_STEP)
      self.broadcast_event(move)
      self.handler(self, move)