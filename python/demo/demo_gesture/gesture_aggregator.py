import time
import numpy as np
from demo.detector import Detector, DetectorEvent
from utils.logger import logger

class GestureAggregator(Detector):
  def __init__(self, name:str, detectors_name:list[str], gestures:list, devices:list=None, handler=None):
    super(GestureAggregator, self).__init__(name=name, handler=handler)
    self.detectors_name = detectors_name
    self.gestures = [self.Gesture(**gesture_config) for gesture_config in gestures]

  def handle_detector_event(self, event:DetectorEvent):
    if event.detector in self.detectors_name:
      for gesture in self.gestures:
        if gesture.update(event.data):
          logger.info(f'Trigger [{event.data}]')

  class Gesture():
    def __init__(self, name, events:list[str], window_time:float=1.0, min_trigger_interval:float=1.0, keep_order=False):
      self.name = name
      self.events = events
      self.window_time = window_time
      self.min_trigger_interval = min_trigger_interval
      self.events_timestamp = np.zeros(len(events))
      self.keep_order = keep_order
      self.last_trigger_time = 0

    def update(self, event:str) -> bool:
      if event not in self.events:
        return False
      current_time = time.time()
      self.events_timestamp[self.events.index(event)] = current_time
      if np.max(self.events_timestamp) - np.min(self.events_timestamp) < self.window_time \
         and (not self.keep_order or np.all(self.events_timestamp[:-1] <= self.events_timestamp[1:])) \
         and current_time > self.last_trigger_time + self.min_trigger_interval:
         self.last_trigger_time = current_time
         return True
      return False
