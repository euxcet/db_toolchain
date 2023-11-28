import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../..'))

import time
import argparse
import numpy as np
from sensor import ring_pool, RingConfig
from sensor.glove import glove_pool, GloveConfig
from utils.file_utils import load_json
from demo.detector import Detector, DetectorEvent
from demo.detectors.gesture_detector import GestureDetector
from utils.logger import logger

class GestureAggregator(Detector):
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

  def __init__(self, arguments:dict):
    super(GestureAggregator, self).__init__(arguments=arguments)
    self.gestures = [self.Gesture(**gesture_config) for gesture_config in arguments['gestures']]

  def handle_detector_event(self, event:DetectorEvent):
    # TODO: Add a specific option in the config to select the listening targets
    if event.detector in self.arguments['detectors_name']:
      for gesture in self.gestures:
        if gesture.update(event.data):
          logger.info(f'Trigger [{event.data}]')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-r', '--ring', type=str, default='0D85597D-C82C-E839-E0E0-4776246A6398')
  parser.add_argument('-g', '--glove', type=str, default='127.0.0.1')
  parser.add_argument('-c', '--config', type=str, default='config/ring_only_config.json')
  args = parser.parse_args()

  config = load_json(args.config)

  if 'ring_gesture_detector' in config:
    ring = ring_pool.add_ring(RingConfig(args.ring, 'GestureRing'))
    ring_gesture_detector = GestureDetector(ring, handler=None, arguments=config['ring_gesture_detector'])
  if 'glove_gesture_detector' in config:
    glove = glove_pool.add_glove(GloveConfig(ip=args.glove))
    glove_gesture_detector = GestureDetector(glove, handler=None, arguments=config['glove_gesture_detector'])

  mouse_ring = GestureAggregator(arguments=config['gesture_aggregator'])
  while True:
    time.sleep(1)
