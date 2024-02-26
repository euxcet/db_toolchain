import time
from queue import Queue
from typing_extensions import override
from db_graph.framework.graph import Graph
from db_graph.framework.node import Node

class MockRing(Node):

  OUTPUT_EDGE_DATA = 'data'
  
  def __init__(
      self,
      name: str,
      graph: Graph,
      input_edges: dict[str, str],
      output_edges: dict[str, str],
      data_freq: int,
      send_freq: int,
  ) -> None:
    super(MockRing, self).__init__(
      name=name,
      graph=graph,
      input_edges=input_edges,
      output_edges=output_edges,
    )
    self.data_freq = data_freq
    self.send_freq = send_freq

  @override
  def start(self) -> None:
    self.queue = Queue()
    self.graph.executor.submit(self.generate_data)
    self.graph.executor.submit(self.send_data)

  def generate_data(self) -> None:
    tot = 0
    last_t = time.perf_counter()
    while True:
      if (t := time.perf_counter()) - last_t >= 1.0 / self.data_freq:
        last_t = t
        self.queue.put(tot)
        tot += 1
        time.sleep(0.001)
        
  
  def send_data(self) -> None:
    while True:
      time.sleep(1.0 / self.send_freq)
      while not self.queue.empty():
        data = self.queue.get()
        self.output(self.OUTPUT_EDGE_DATA, data)
