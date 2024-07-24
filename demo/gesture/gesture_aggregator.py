import time
import numpy as np
from typing_extensions import override
from db_graph.framework.graph import Graph
from db_graph.framework.node import Node

class GestureAggregator(Node):

  INPUT_EDGE_EVENT = 'event'
  INPUT_EDGE_TOUCH = 'touch'
  OUTPUT_EDGE_RESULT = 'result'

  def __init__(
      self,
      name: str,
      graph: Graph,
      input_edges: dict[str, str],
      output_edges: dict[str, str],
      gestures: list,
  ) -> None:
    super(GestureAggregator, self).__init__(
      name=name,
      graph=graph,
      input_edges=input_edges,
      output_edges=output_edges,
    )
    self.gestures = [self.Gesture(**gesture_config) for gesture_config in gestures]

  @override
  def start(self):
    ...

  def handle_input_edge_touch(self, event: str, timestamp: float) -> None:
    print(event)

  def handle_input_edge_event(self, event: str, timestamp: float) -> None:
    for gesture in self.gestures:
      if gesture.update(event):
        self.log_info(gesture.name)
        self.output(self.OUTPUT_EDGE_RESULT, gesture.name)

  class Gesture():
    def __init__(
        self,
        name: str,
        events: list[str],
        window_time: float = 1.0,
        min_trigger_interval: float = 1.0,
        keep_order: bool = False,
    ) -> None:
      self.name = name
      self.events = events
      self.window_time = window_time
      self.min_trigger_interval = min_trigger_interval
      self.events_timestamp = np.zeros(len(events))
      self.keep_order = keep_order
      self.last_trigger_time = 0

    def update(self, event: str) -> bool:
      if event not in self.events:
        return False
      current_time = time.time()
      self.events_timestamp[self.events.index(event)] = current_time
      if np.max(self.events_timestamp) - np.min(self.events_timestamp) < self.window_time \
         and (not self.keep_order or np.all(self.events_timestamp[:-1] <= self.events_timestamp[1:])) \
         and current_time > self.last_trigger_time + self.min_trigger_interval:
         self.last_trigger_time = current_time
         return True
      return False
