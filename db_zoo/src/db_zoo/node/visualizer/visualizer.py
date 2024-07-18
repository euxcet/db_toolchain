import struct
import numpy as np
from typing import Any
from vispy import app
from typing_extensions import override
from db_graph.framework.graph import Graph
from db_graph.framework.node import Node
from collections import deque
from ...utils.plotter import MultiPlotter

class Visualizer(Node):

  INPUT_EDGE_DATA = 'data'
  
  def __init__(
      self,
      name: str,
      graph: Graph,
      input_edges: dict[str, str],
      output_edges: dict[str, str],
  ) -> None:
    super(Visualizer, self).__init__(
      name=name,
      graph=graph,
      input_edges=input_edges,
      output_edges=output_edges,
    )
    self.last_timestamp = 0
    self.plotter = MultiPlotter()
    self.buffer = deque(maxlen=2000)
    self.plotter.add_single_line_plotter(
      row=0,
      col=0,
      size=200,
      data_buffer=self.buffer,
      x_range=(0, 200),
      y_range=(-200, 200)
    )

  @override
  def start(self) -> None:
    ...

  @override
  def block(self) -> None:
    self.plotter.start(1 / 30)
    app.run()

  def handle_input_edge_data(self, data: Any, timestamp: float) -> None:
    print(data)
    self.buffer.append(data)