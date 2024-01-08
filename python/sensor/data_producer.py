import queue
from typing import Any, Callable, TypeVar
from abc import ABCMeta, abstractmethod
from stream import stream_manager, Stream
from utils.logger import logger

T = TypeVar('T')

class DataProducer(metaclass=ABCMeta):
  def __init__(self) -> None:
    self.streams:dict[str, Stream] = {}
    self.initialize_stream()

  @property
  @abstractmethod
  def name(self) -> str:
    pass

  @property
  @abstractmethod
  def stream_names(self) -> list[str]:
    pass

  # stream
  def initialize_stream(self) -> None:
    for stream_name in self.stream_names:
      self.create_stream(stream_name)

  def get_stream_name(self, local_stream_name:str) -> str:
    return self.name + '_' + str(local_stream_name)

  def get_stream(self, local_stream_name:str) -> Stream:
    return self.streams[local_stream_name]

  def create_stream(self, local_stream_name:str) -> Stream:
    self.streams[local_stream_name] = stream_manager.add_stream(Stream(self.get_stream_name(local_stream_name)))
    return self.streams[local_stream_name]

  def produce_data(self, local_stream_name:str, data:Any, timestamp:float=None):
    self.streams[local_stream_name].put(data, timestamp)

  # log
  def log_info(self, msg, *args, **kwargs):
    logger.info(f'[{self.name}] ' + msg, *args, **kwargs)

  def log_warning(self, msg, *args, **kwargs):
    logger.warning(f'[{self.name}] ' + msg, *args, **kwargs)

  def log_error(self, msg, *args, **kwargs):
    logger.error(f'[{self.name}] ' + msg, *args, **kwargs)

  def log_critical(self, msg, *args, **kwargs):
    logger.critical(f'[{self.name}] ' + msg, *args, **kwargs)

