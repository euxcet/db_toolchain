from typing import Any
from abc import ABC, abstractmethod

from .edge import Edge
from .edge_manager import EdgeManager
from ..utils.logger import logger
from ..utils.register import Register
from ..utils.counter import Counter

class Node(ABC):

  HANDLE_FUNCTION_PREFIX = 'handle_input_edge_'
  CLASS_NOT_REQUIRED_TO_REGISTER = ['Node', 'TorchNode', 'Device']

  '''
    input_edges: local_key -> edge_name
    outptu_edges: local_key -> edge_name
  '''
  def __init__(
      self,
      name: str,
      edge_manager: EdgeManager,
      input_edges: dict[str, str],
      output_edges: dict[str, str],
  ) -> None:
    self.name = name
    self.edge_manager = edge_manager
    self.input_edge_names = input_edges
    self.output_edge_names = self._complete_output_edge_names(output_edges)
    self.output_edge: dict[str, Edge] = {}
    for local_key, edge_name in self.input_edge_names.items():
      self.edge_manager.bind_edge(edge_name, getattr(self, f'{self.HANDLE_FUNCTION_PREFIX}{local_key}'))
    for local_key, edge_name in self.output_edge_names.items():
      self.output_edge[local_key] = self.edge_manager.add_edge(Edge(edge_name))
    self.counter = Counter()

  def output(self, edge_name:str, data:Any) -> None:
    self.output_edge[edge_name].put(data)

  def __init_subclass__(cls) -> None:
    if cls.__name__ not in Node.CLASS_NOT_REQUIRED_TO_REGISTER:
      node_register.register(cls.__name__, cls)
    return super().__init_subclass__()

  def _default_stream_name(self, stream: str) -> str:
    return f'{self.name}_{stream}'
  
  def _complete_edge_names(self, names: dict[str, str], prefix: str) -> dict[str, str]:
    for variable in dir(self):
      if variable.startswith(prefix):
        names[getattr(self, variable)] = self._default_stream_name(getattr(self, variable))
    return names

  def _complete_output_edge_names(self, names: dict[str, str]) -> dict[str, str]:
    return self._complete_edge_names(names, 'OUTPUT_EDGE_')

  @property
  def _log_prefix(self) -> str:
    return f'(NODE {self.name}) '

  def log_info(self, msg: Any, *args: Any, **kwargs: Any):
    logger.info(self._log_prefix + str(msg), *args, **kwargs)

  def log_warning(self, msg: Any, *args: Any, **kwargs: Any):
    logger.warning(self._log_prefix + str(msg), *args, **kwargs)

  def log_error(self, msg: Any, *args: Any, **kwargs: Any):
    logger.error(self._log_prefix + str(msg), *args, **kwargs)

  def log_critical(self, msg: Any, *args: Any, **kwargs: Any):
    logger.critical(self._log_prefix + str(msg), *args, **kwargs)

node_register = Register[Node]()