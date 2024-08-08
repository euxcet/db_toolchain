import time
import wave
import numpy as np
from threading import Thread
from typing_extensions import override
from db_graph.utils.counter import Counter
from db_graph.framework.graph import Graph
from db_graph.framework.node import Node
import struct

class DroneController(Node):
  INPUT_EDGE_RESPONSE = 'response'
  OUTPUT_EDGE_COMMAND = 'command'

  def __init__(
      self,
      name: str,
      graph: Graph,
      input_edges: dict[str, str],
      output_edges: dict[str, str],
  ) -> None:
    super().__init__(
      name=name,
      graph=graph,
      input_edges=input_edges,
      output_edges=output_edges,
    )

  @override
  def start(self):
    Thread(target=self.perform_action).start()

  def perform_action(self):
    while True:
        action = input()
        print('Action:', action)
        self.output(self.OUTPUT_EDGE_COMMAND, action.strip())

  def handle_input_edge_response(self, data, timestamp: float) -> None:
    print(data)
