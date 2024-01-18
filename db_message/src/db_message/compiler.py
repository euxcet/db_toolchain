from pathlib import Path
from abc import ABC, abstractmethod
from .register import Register

# TODO: compile multiple files
class Compiler():
  def __init__(self) -> None:
    pass

  @abstractmethod
  def compile(self) -> None:
    ...

  def __init_subclass__(cls) -> None:
    compiler_register.register(cls.__name__, cls)
    return super().__init_subclass__()

compiler_register = Register[Compiler]()