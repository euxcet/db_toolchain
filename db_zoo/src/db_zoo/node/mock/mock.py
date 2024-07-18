import time
import random
from typing_extensions import override
from db_graph.framework.graph import Graph
from db_graph.framework.node import Node

class Mock(Node):

  OUTPUT_EDGE_DATA = 'data'
  
  def __init__(
      self,
      name: str,
      graph: Graph,
      input_edges: dict[str, str],
      output_edges: dict[str, str],
  ) -> None:
    super(Mock, self).__init__(
      name=name,
      graph=graph,
      input_edges=input_edges,
      output_edges=output_edges,
    )

  @override
  def start(self) -> None:
    self.graph.executor.submit(self.generate_data)

  def generate_data(self) -> None:
    while True:
        time.sleep(0.05)
        self.output(self.OUTPUT_EDGE_DATA, random.randint(-127, 128))
