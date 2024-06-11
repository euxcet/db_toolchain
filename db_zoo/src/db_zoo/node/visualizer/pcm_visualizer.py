import struct
import numpy as np
from typing import Any
from vispy import app
from typing_extensions import override
from db_graph.framework.graph import Graph
from db_graph.framework.node import Node
from collections import deque
from multiprocessing import Process
from threading import Thread
from ...utils.plotter import MultiPlotter

class PcmVisualizer(Node):

  INPUT_EDGE_IN_DATA = 'in_data'
  
  def __init__(
      self,
      name: str,
      graph: Graph,
      input_edges: dict[str, str],
      output_edges: dict[str, str],
  ) -> None:
    super(PcmVisualizer, self).__init__(
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
      size=2000,
      data_buffer=self.buffer,
      x_range=(0, 2000),
      y_range=(-2000, 2000)
    )

  @override
  def start(self) -> None:
    ...

  @override
  def block(self) -> None:
    self.plotter.start(1 / 30)
    app.run()

  def handle_input_edge_in_data(self, data: Any, timestamp: float) -> None:
    length, seq, bytes = data
    for i in range(0, len(bytes), 2):
      self.buffer.append(struct.unpack('<h', bytes[i:i + 2]))