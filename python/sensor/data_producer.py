from typing import Any, Callable
import queue
from abc import ABCMeta, abstractmethod
from utils.logger import logger

class DataProducer(metaclass=ABCMeta):
  def __init__(self, event_queue:queue.Queue):
    self.event_queue = event_queue

  @property
  @abstractmethod
  def name(self) -> str:
    pass

  def produce_event(self, event:Any):
    if self.event_queue is not None:
      self.event_queue.put_nowait(event)

  def log_info(self, msg, *args, **kwargs):
    logger.info(f'[{self.name}] ' + msg, *args, **kwargs)

  def log_warning(self, msg, *args, **kwargs):
    logger.warning(f'[{self.name}] ' + msg, *args, **kwargs)

  def log_error(self, msg, *args, **kwargs):
    logger.error(f'[{self.name}] ' + msg, *args, **kwargs)

  def log_critical(self, msg, *args, **kwargs):
    logger.critical(f'[{self.name}] ' + msg, *args, **kwargs)

