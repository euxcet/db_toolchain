import numpy as np
from demo.detector import Detector, DetectorEvent

class MouseRing(Detector):
  def __init__(self, name:str, trajectory_detector_name:str, gesture_detector_name:str, cursor_scale:float,
               cursor_initial_x:int, cursor_initial_y:int, control_cursor:bool=False, devices:list=None, handler=None):
    super(MouseRing, self).__init__(name=name, handler=handler)
    self.trajectory_detector_name = trajectory_detector_name
    self.gesture_detector_name = gesture_detector_name
    self.cursor_scale = cursor_scale
    self.cursor_initial_x = cursor_initial_x
    self.cursor_initial_y = cursor_initial_y
    self.control_cursor = control_cursor
    self.counter.print_interval = 100
    self.current_move = np.zeros(2)
    if control_cursor:
      from pynput.mouse import Controller
      self.mouse = Controller()

  def reset(self):
    if self.control_cursor:
      self.mouse.position = (self.cursor_initial_x, self.cursor_initial_y)
  
  def handle_detector_event(self, event:DetectorEvent):
    if event.detector == self.trajectory_detector_name:
      if self.control_cursor:
        self.mouse.move(event.data[0] * self.cursor_scale, event.data[1] * self.cursor_scale)
    if event.detector == self.gesture_detector_name and self.control_cursor:
      from pynput.mouse import Button
      if event.data == 'click':
        self.mouse.click(Button.left)
      elif event.data == 'double_click':
        self.reset()