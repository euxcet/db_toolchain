import queue
from enum import Enum
from typing import Any
from abc import ABCMeta, abstractmethod
from sensor.data_producer import DataProducer

class DeviceLifeCircleEvent(Enum):
  on_create = 0
  on_pair = 1
  on_connect = 2
  on_disconnect = 3
  on_error = 4

class Device(DataProducer, metaclass=ABCMeta):
  def __init__(self) -> None:
    super(Device, self).__init__()

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