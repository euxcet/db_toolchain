import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../..'))

import time
import argparse
from sensor import ring_pool, RingConfig
from utils.file_utils import load_json
from demo.detector import Detector, DetectorEvent
from demo.detectors.trajectory_detector import TrajectoryDetector
from demo.detectors.gesture_detector import GestureDetector
import numpy as np

class MouseRing(Detector):
  def __init__(self, arguments:dict):
    super(MouseRing, self).__init__(arguments=arguments)
    self.counter.print_interval = 100
    self.current_move = np.zeros(2)
    if arguments['control_cursor']:
      from pynput.mouse import Controller
      self.mouse = Controller()

  def reset(self):
    if self.arguments['control_cursor']:
      self.mouse.position = (self.arguments['cursor_initial_point_x'], self.arguments['cursor_initial_point_y'])
  
  def handle_detector_event(self, event:DetectorEvent):
    if event.detector == self.arguments['trajectory_detector_name']:
      if self.arguments['control_cursor']:
        scale = self.arguments['cursor_scale']
        self.mouse.move(event.data[0] * scale, event.data[1] * scale)
    if event.detector == self.arguments['gesture_detector_name']:
      if event.data != 'move':
        print(event.data)
      if self.arguments['control_cursor']:
        from pynput.mouse import Button
        if event.data == 'click':
          self.mouse.click(Button.left)
        elif event.data == 'double_click':
          self.reset()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-r', '--ring', type=str, default='0D85597D-C82C-E839-E0E0-4776246A6398')
  parser.add_argument('-c', '--config', type=str, default='config/config.json')
  args = parser.parse_args()

  ring = ring_pool.add_ring(RingConfig(args.ring, 'MouseRing'))
  config = load_json(args.config)
  trajectory_detector = TrajectoryDetector(ring, handler=None, arguments=config['trajectory_detector'])
  gesture_detector = GestureDetector(ring, handler=None, arguments=config['gesture_detector'])
  mouse_ring = MouseRing(arguments=config['mouse_ring'])
  while True:
    time.sleep(1)
