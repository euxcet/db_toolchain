from typing import Any
from typing_extensions import override
from db_graph.framework.graph import Graph
from db_graph.framework.node import Node

class Printer(Node):

  INPUT_EDGE_IN_DATA = 'in_data'
  OUTPUT_EDGE_OUT_DATA = 'out_data'
  
  def __init__(
      self,
      name: str,
      graph: Graph,
      input_edges: dict[str, str],
      output_edges: dict[str, str],
  ) -> None:
    super(Printer, self).__init__(
      name=name,
      graph=graph,
      input_edges=input_edges,
      output_edges=output_edges,
    )
    self.last_timestamp = 0
    self.counter.print_interval = 500

  @override
  def start(self) -> None:
    ...

  def handle_input_edge_in_data(self, data: Any, timestamp: float) -> None:
    self.counter.count(print_dict={'name': 'print'}, enable_print=True)
    # print('data', data, 'timestamp', timestamp)
    self.last_timestamp = timestamp
    self.output(self.OUTPUT_EDGE_OUT_DATA, data)