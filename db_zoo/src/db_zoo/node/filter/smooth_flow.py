import time
import select
from typing import Any
from queue import Queue
from typing_extensions import override
from db_graph.framework.graph import Graph
from db_graph.framework.node import Node


class SmoothFlow(Node):

  INPUT_EDGE_IN_DATA = 'in_data'
  OUTPUT_EDGE_OUT_DATA = 'out_data'
  
  def __init__(
      self,
      name: str,
      graph: Graph,
      input_edges: dict[str, str],
      output_edges: dict[str, str],
      start_freq: float = 400,
      start_size: int = 100,
      min_size: int = 20,
      max_size: int = 150,
      calc_freq_gap: int = 200,
  ) -> None:
    super(SmoothFlow, self).__init__(
      name=name,
      graph=graph,
      input_edges=input_edges,
      output_edges=output_edges,
    )
    self.start_freq = start_freq
    self.freq = self.start_freq
    self.buffer: Queue = Queue()
    self.start_size = start_size
    self.min_size = min_size
    self.max_size = max_size
    self.calc_freq_gap = calc_freq_gap
    self.started = False
    self.counter.print_interval = 500

  @override
  def start(self) -> None:
    ...

  def send_data(self) -> None:
    last_t = time.perf_counter()
    while True:
      t = time.perf_counter()
      if t - last_t >= 1.0 / self.freq:
        # buffer_size = self.buffer.qsize()
        print(t - last_t, 1.0 / self.freq)
        self.counter.count(enable_print=True)
        last_t = t
        # if buffer_size >= self.min_size and buffer_size <= self.max_size:
        # self.output(self.OUTPUT_EDGE_OUT_DATA, self.buffer.get())
        # elif buffer_size > self.max_size:
        #   # freq is too small
        #   # calculate the freq
        #   ...
        # elif buffer_size < self.min_size:
        #   # freq is too big
        #   # calculate the freq
        #   ...
        time.sleep(0.0001)

  def handle_input_edge_in_data(self, data: Any, timestamp: float) -> None:
    self.graph.executor.submit(self.send_data)
    # self.counter.count(enable_print=True)
    self.buffer.put(data)
    if not self.started and self.buffer.qsize() >= self.start_size:
      self.started = True
      self.graph.executor.submit(self.send_data)
