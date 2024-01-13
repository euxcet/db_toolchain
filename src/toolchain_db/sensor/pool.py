from abc import ABCMeta, abstractmethod
from sensor.device import Device
from typing import Any

class Pool(metaclass=ABCMeta):
  def __init__(self): ...
  
  @abstractmethod
  def add_device(self, config:Any) -> Device: ...

  @abstractmethod
  def get_device(self, config:Any) -> None: ...

  @abstractmethod
  def keep_alive_strategy(self) -> None: ...