import time
from queue import Queue
from typing import TypeVar, Generic, Any, Callable

T = TypeVar('T')

class Stream(Generic[T]):
  def __init__(self, name:str, direct:bool=True) -> None:
    self.name = name
    self.direct = direct
    self.queue = Queue[tuple[T, float]]()
    self.handlers:list[Callable] = []

  def get(self, block:bool=True, timeout:float=None) -> tuple[T, float]:
    try:
      return self.queue.get(block, timeout)
    except:
      return None

  def get_no_wait(self) -> tuple[T, float]:
    try:
      return self.queue.get_nowait()
    except:
      return None

  def put(self, data:T, timestamp:float=None) -> None:
    timestamp = time.time() if timestamp is None else timestamp
    if self.direct:
      for handler in self.handlers:
        handler(data, timestamp)
    else:
      self.queue.put((data, timestamp))

  def bind(self, handler:Callable) -> None:
    self.handlers.append(handler)

  def unbind(self, handler:Callable) -> None:
    self.handlers.remove(handler)