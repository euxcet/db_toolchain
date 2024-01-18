from pathlib import Path
from ..compiler import Compiler

# do nothing
class TextCompiler():
  def __init__(self) -> None:
    super(TextCompiler, self).__init__()

  def compile(self) -> None:
    ...