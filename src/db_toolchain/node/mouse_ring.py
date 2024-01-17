import numpy as np
from ..framework.torch_node import Detector

class MouseRing(Detector):
  def __init__(self, name:str, input_streams:dict[str, str], output_streams:dict[str, str], cursor_scale:float,
               cursor_initial_x:int, cursor_initial_y:int, control_cursor:bool=False):
    super(MouseRing, self).__init__(name=name, input_streams=input_streams, output_streams=output_streams)
    self.cursor_scale = cursor_scale
    self.cursor_initial_x = cursor_initial_x
    self.cursor_initial_y = cursor_initial_y
    self.control_cursor = control_cursor
    self.counter.print_interval = 100
    self.current_move = np.zeros(2)
    if control_cursor:
      from pynput.mouse import Controller
      self.mouse = Controller()

  def handle_input_stream_touch_state(self, data:str, timestamp:float) -> None:
    if self.control_cursor:
      from pynput.mouse import Button
      if data == 'click':
        self.mouse.click(Button.left)
      elif data == 'double_click':
        self.mouse.position = (self.cursor_initial_x, self.cursor_initial_y)

  def handle_input_stream_trajectory(self, data:tuple[float, float], timestamp:float) -> None:
    if self.control_cursor:
      self.mouse.move(data[0] * self.cursor_scale, data[1] * self.cursor_scale)
