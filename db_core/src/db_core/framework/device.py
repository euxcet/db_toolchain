import queue
from enum import Enum
from typing import Any
from abc import ABCMeta, abstractmethod
from .node import Node

class DeviceLifeCircleEvent(Enum):
  on_create = 0
  on_pair = 1
  on_connect = 2
  on_disconnect = 3
  on_error = 4

class Device(Node, metaclass=ABCMeta):
  def __init__(
      self,
      name: str,
      input_edges: dict[str, str],
      output_edges: dict[str, str],
  ) -> None:
    super(Device, self).__init__(
      name=name,
      input_edges=input_edges,
      output_edges=output_edges
    )

  # lifecycle callbacks
  @abstractmethod
  def on_pair(self) -> None: ...

  @abstractmethod
  def on_connect(self) -> None: ...

  @abstractmethod
  def on_disconnect(self) -> None: ...

  @abstractmethod
  def on_error(self) -> None: ...

  # active control
  @abstractmethod
  def connect(self) -> None: ...

  @abstractmethod
  def disconnect(self) -> None: ...

  @abstractmethod
  def reconnect(self) -> None: ...