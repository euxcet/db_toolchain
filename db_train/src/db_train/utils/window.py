from __future__ import annotations

import math
import numpy as np
from typing import TypeVar, Generic

_T = TypeVar('_T')

class Window(Generic[_T]):
  def __init__(self, window_length: int, window: list[_T] = None):
    self.window_length = window_length
    self.window:list[_T] = [] if window is None else window
    self.push_count = 0
  
  def push(self, data:_T):
    self.push_count += 1
    self.window.append(data)
    if len(self.window) > self.window_length:
      self.window.pop(0)

  def clear(self):
    self.window.clear()

  def first(self) -> _T:
    return self.window[0]

  def last(self) -> _T:
    return self.window[-1]

  def get(self, index: int) -> _T:
    return self.window[index]

  def head(self, length: int) -> Window[_T]:
    return Window[_T](self.window_length, self.window[:length])

  def tail(self, length: int) -> Window[_T]:
    return Window[_T](self.window_length, self.window[-length:])

  def capacity(self):
    return len(self.window)

  def empty(self):
    return len(self.window) == 0

  def full(self):
    return len(self.window) == self.window_length

  def sum(self, func: function = lambda x:x):
    return sum(map(func, self.window))

  def count(self, func: function = lambda x:x) -> int:
    return len(list(filter(lambda x:x == True, map(func, self.window))))

  def any(self, func: function = lambda x:x) -> bool:
    return any(map(func, self.window))

  def all(self, func: function = lambda x:x) -> bool:
    return all(map(func, self.window))

  def map(self, func: function = lambda x:x) -> Window:
    return Window(self.window_length, list(map(func, self.window)))

  def argmax(self) -> tuple[int, _T]:
    if self.capacity() == 0:
      return 0
    index = 0
    value = self.window[0]
    for i, v in enumerate(self.window):
      if v > value:
        value = v
        index = i
    return index, value

  def to_numpy(self):
    try:
      return np.array([d.to_numpy() for d in self.window])
    except:
      return np.array(self.window)

  def to_numpy_float(self):
    return self.to_numpy().astype('float32')

  def pad(self):
    self.window += self.window[-1:] * (self.window_length - len(self.window))
    return self

  def __len__(self) -> int:
    return len(self.window)
