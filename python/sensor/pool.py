from abc import ABCMeta, abstractmethod
from sensor.device import Device
from typing import Any

class Pool(metaclass=ABCMeta):
  def __init__(self):
    pass
  
  @abstractmethod
  def add_device(self, config:Any) -> Device:
    pass

  @abstractmethod
  def get_device(self, config:Any) -> None:
    pass

  @abstractmethod
  def keep_alive_strategy(self) -> None:
    pass