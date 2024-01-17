from typing import TypeVar, Generic, Any
from .logger import logger

T = TypeVar('T')

class Register(Generic[T]):
  def __init__(self) -> None:
    self.items:dict[str, T] = {}

  def register(self, name:str, item:T) -> None:
    if name not in self.items:
      self.items[name] = item
      logger.info(f'Register node {name}.')
    else:
      logger.warning(f'{name} has been registered.')

  def instance(self, name:str, kwargs:dict[str, Any], ignore_keys:list[str]=[]) -> None:
    for key in ignore_keys:
      try:
        kwargs.pop(key)
      except:
        pass
    if name not in self.items:
      logger.error(f'{name} has not been registered.')
    return self.items[name](**kwargs)
  
  def __len__(self) -> int:
    return len(self.items)