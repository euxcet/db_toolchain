from __future__ import annotations

import re
import copy
from pathlib import Path

class Hierarchy():
  def __init__(
      self,
      parent: Hierarchy | None = None,
      hierarchy: list[str] = [],
  ) -> None:
    self.parent = parent
    self.children = dict()
    self.hierarchy = hierarchy
    self.files: list[Path] = []
    self.name = 'ROOT' if len(self.hierarchy) == 0 else self.hierarchy[-1]

  def create_child(self, child_name: str) -> Hierarchy:
    if child_name in self.children:
      return self.children[child_name]
    child_hierarchy = copy.deepcopy(self.hierarchy)
    child_hierarchy.append(child_name)
    child = Hierarchy(parent=self, hierarchy=child_hierarchy)
    self.children[child_name] = child
    return child

  def get_file(self, pattern: str) -> Path:
    for file in self.files:
      if re.search(pattern, file.name) != None:
        return file
    return None

  def get_hierarchy(self, layer: int) -> str:
    return self.hierarchy[layer]

  def __str__(self) -> str:
    return '/'.join(self.hierarchy)

  def __repr__(self) -> str:
    return '/'.join(self.hierarchy)

  def __lt__(self, other: Hierarchy) -> bool:
    return str(self.hierarchy) < str(other.hierarchy)

# TODO: blacklist
class FileLoader():
  def __init__(self, root: Path):
    self.root = root
    self.hierarchies: set[Hierarchy] = set()

  def _walk_dir(self, dir: Path, hierarchy: Hierarchy) -> list[tuple[Hierarchy, Path]]:
    paths = sorted(
        Path(dir).iterdir(),
        key = lambda path: (path.is_file(), path.name.lower()),
    )
    files = []
    for path in paths:
      if path.name.startswith("."):
        continue
      if path.is_dir():
        files.extend(self._walk_dir(path, hierarchy.create_child(path.name)))
      else:
        file_hierarchy = hierarchy
        for hier in path.name.split('_')[:-1]:
          file_hierarchy = file_hierarchy.create_child(hier)
        file_hierarchy.files.append(path)
        files.append((file_hierarchy, path))
    return files

  def load(self) -> tuple[list[tuple[Hierarchy, Path]], Hierarchy]:
    self.root_hierarchy = Hierarchy()
    self.files: list[tuple[Hierarchy, Path]] = self._walk_dir(self.root, self.root_hierarchy)
    for hier, _ in self.files:
      self.hierarchies.add(hier)
    return self.files, self.root_hierarchy