import time
import numpy as np
from typing_extensions import override
from db_graph.framework.graph import Graph
from db_graph.framework.node import Node
from db_graph.data.imu_data import IMUData

class DailyRecorder(Node):
  INPUT_EDGE_EVENT = 'event'

  def __init__(
      self,
      name: str,
      graph: Graph,
      input_edges: dict[str, str],
      output_edges: dict[str, str],
      user: str,
  ) -> None:
    super(DailyRecorder, self).__init__(
      name=name,
      graph=graph,
      input_edges=input_edges,
      output_edges=output_edges,
    )
    self.user = user

  @override
  def start(self):
    # create file
    ...

  def handle_input_edge_imu(self, data: IMUData, timestamp: float) -> None:
    print(data, timestamp)
