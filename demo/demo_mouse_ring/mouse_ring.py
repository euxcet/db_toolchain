import numpy as np
from db_graph.framework.graph import Graph
from db_graph.framework.node import Node

class MouseRing(Node):

  INPUT_EDGE_touch_state = 'touch_state'
  INPUT_EDGE_TRAJECTORY = 'trajectory'
  OUTPUT_EDGE_RESULT = 'result'

  def __init__(
      self,
      name: str,
      graph: Graph,
      input_edges: dict[str, str],
      output_edges: dict[str, str],
      cursor_scale: float = 10,
      cursor_initial_x: int = 300,
      cursor_initial_y: int = 300,
      control_cursor: bool = False,
  ) -> None:
    super(MouseRing, self).__init__(
      name=name,
      graph=graph,
      input_edges=input_edges,
      output_edges=output_edges,
    )
    self.cursor_scale = cursor_scale
    self.cursor_initial_x = cursor_initial_x
    self.cursor_initial_y = cursor_initial_y
    self.control_cursor = control_cursor
    self.counter.print_interval = 100
    self.current_move = np.zeros(2)
    if control_cursor:
      from pynput.mouse import Controller
      self.mouse = Controller()

  def handle_input_edge_touch_state(self, data:str, timestamp:float) -> None:
    if self.control_cursor:
      from pynput.mouse import Button
      if data == 'click':
        self.mouse.click(Button.left)
      elif data == 'double_click':
        self.mouse.position = (self.cursor_initial_x, self.cursor_initial_y)

  def handle_input_edge_trajectory(self, data:tuple[float, float], timestamp:float) -> None:
    if self.control_cursor:
      self.mouse.move(data[0] * self.cursor_scale, data[1] * self.cursor_scale)
