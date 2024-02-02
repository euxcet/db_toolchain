import time
from typing import Callable
from threading import Thread

from .edge import Edge, PlaceHolderEdge

class EdgeManager():
  def __init__(self) -> None:
    self.edges:dict[str, Edge] = {}
    Thread(target=self.distribute).start()

  @property
  def direct_edge_count(self) -> int:
    return list(map(lambda x: x.direct, self.edges.values())).count(True)

  @property
  def indirect_edge_count(self) -> int:
    return len(self.edges) - self.direct_edge_count

  def get_edge(self, name:str) -> Edge:
    name = name.lower()
    return self.edges[name]

  def add_edge(self, stream:Edge) -> Edge:
    name = stream.name.lower()
    if name in self.edges:
      if type(self.edges[name]) is PlaceHolderEdge:
        stream.take_over(self.edges[name])
      else:
        raise Exception()
    self.edges[name.lower()] = stream
    return stream

  def bind_edge(self, name:str, handler:Callable) -> None:
    if name.lower() not in self.edges:
      self.add_edge(PlaceHolderEdge(name))
    self.edges[name.lower()].bind(handler)

  def unbind_stream(self, name:str, handler:Callable) -> None:
    name = name.lower()
    self.edges[name].unbind(handler)

  def distribute(self):
    while True:
      if self.indirect_edge_count > 0:
        for name, stream in self.edges.items():
          data = stream.get_no_wait()
          if data is not None:
            for handler in self.edges[name].handlers:
              handler(*data)
        time.sleep(0.001)
      else:
        time.sleep(0.01)

edge_manager = EdgeManager()
