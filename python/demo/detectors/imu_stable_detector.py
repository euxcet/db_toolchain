import numpy as np
from utils.window import Window
from demo.detector import Detector, DetectorEvent
from sensor.basic_data import IMUData
from sensor import Ring, RingEvent, RingEventType
from sensor.glove import Glove, GloveEvent, GloveEventType
from sensor.glove_data import GloveIMUJointName

class IMUStableDetector(Detector):
  def __init__(self, name:str, device:str, devices:dict, imu_window_length:int, execute_interval:int, handler=None):
    super(IMUStableDetector, self).__init__(name=name, device=devices[device], handler=handler)
    self.imu_window_length = imu_window_length
    self.stable_window = Window[IMUData](self.imu_window_length)
    self.counter.execute_interval = execute_interval

  def detect(self, data):
    self.stable_window.push(np.linalg.norm(data[:3]) < 0.1)
    if self.counter.count(enable_print=True, print_fps=True) and self.stable_window.full():
      self.broadcast_event(self.stable_window.all())

  def handle_detector_event(self, event:DetectorEvent):
    pass

  def handle_ring_event(self, device, event:RingEvent):
    if event.event_type == RingEventType.imu:
      self.detect(event.data.to_numpy())

  def handle_glove_event(self, device, event:GloveEvent):
    if event.event_type == GloveEventType.pose:
      self.detect(event.data.get_imu_data(GloveIMUJointName.INDEX_INTERMEDIATE).to_numpy())
