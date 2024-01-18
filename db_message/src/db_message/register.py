from typing import TypeVar, Generic, Any

T = TypeVar('T')

class Register(Generic[T]):
  def __init__(self) -> None:
    self.items:dict[str, T] = {}

  def register(self, name:str, item:T) -> None:
    if name not in self.items:
      self.items[name] = item

  def instance(self, name:str, kwargs:dict[str, Any]={}, ignore_keys:list[str]=[]) -> None:
    for key in ignore_keys:
      try:
        kwargs.pop(key)
      except:
        pass
    if name not in self.items:
      raise KeyError(f'{name} is not registered yet.')
    return self.items[name](**kwargs)
  
  def __len__(self) -> int:
    return len(self.items)