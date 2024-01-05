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
  def __init__(self, event_queue:queue.Queue):
    super(Device, self).__init__(event_queue)
    pass

  # lifecycle callbacks
  @abstractmethod
  def on_pair(self):
    pass

  @abstractmethod
  def on_connect(self):
    pass

  @abstractmethod
  def on_disconnect(self):
    pass

  @abstractmethod
  def on_error(self):
    pass

  # active control
  @abstractmethod
  def connect(self):
    pass

  @abstractmethod
  def disconnect(self):
    pass

  @abstractmethod
  def reconnect(self):
    pass