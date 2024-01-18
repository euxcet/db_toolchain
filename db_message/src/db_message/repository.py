from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

class Repository(ABC):
  def __init__(
      self,
      name: str,
      local: Path,
      remote: str,
  ) -> None:
    self.name = name
    self.local = local
    self.remote = remote

  def to_row(self) -> list[str]:
    return [self.name, self.__class__.__name__, str(self.local), self.remote]

  @property
  @abstractmethod
  def info(self) -> list[str]:
    ...

  @abstractmethod
  def clone(self, url: str, local: Path) -> Repository:
    ...

  @abstractmethod
  def update(self) -> Repository:
    ...

  @abstractmethod
  def check(self, path: Path) -> bool:
    ...