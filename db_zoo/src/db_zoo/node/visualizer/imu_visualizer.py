import struct
import numpy as np
from typing import Any
from vispy import app
from typing_extensions import override
from db_graph.framework.graph import Graph
from db_graph.framework.node import Node
from collections import deque
from db_graph.data.imu_data import IMUData
from ...utils.plotter import MultiPlotter

class ImuVisualizer(Node):

  INPUT_EDGE_IMU = 'imu'
  
  def __init__(
      self,
      name: str,
      graph: Graph,
      input_edges: dict[str, str],
      output_edges: dict[str, str],
  ) -> None:
    super(ImuVisualizer, self).__init__(
      name=name,
      graph=graph,
      input_edges=input_edges,
      output_edges=output_edges,
    )
    self.last_timestamp = 0
    self.plotter = MultiPlotter()
    self.buffer = [deque(maxlen=400) for _ in range(6)]
    for i in range(6):
        self.plotter.add_single_line_plotter(
            row=i // 3,
            col=i % 3,
            size=400,
            data_buffer=self.buffer[i],
            x_range=(0, 400),
            y_range=(-20, 20)
        )

  @override
  def start(self) -> None:
    ...

  @override
  def block(self) -> None:
    self.plotter.start(1 / 40)
    app.run()

  def handle_input_edge_imu(self, data: IMUData, timestamp: float) -> None:
    self.buffer[0].append(data.acc_x)
    self.buffer[1].append(data.acc_y)
    self.buffer[2].append(data.acc_z)
    self.buffer[3].append(data.gyr_x)
    self.buffer[4].append(data.gyr_y)
    self.buffer[5].append(data.gyr_z)