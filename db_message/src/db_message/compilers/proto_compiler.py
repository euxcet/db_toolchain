from pathlib import Path
import subprocess
from ..compiler import Compiler

class ProtoCompiler(Compiler):
  def __init__(self) -> None:
    super(ProtoCompiler, self).__init__()

  def compile(self, file: Path) -> None:
    subprocess.run(
      [
        'protoc',
        '--python_out=.',
        '--pyi_out=.',
        '-I',
        '.',
        str(file)
      ]
    )
