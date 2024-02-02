from typing import Any
from pathlib import Path
import json
import readline
from rich.console import Console
from rich import print

def load_json(path: Path) -> Any:
  with path.open(mode='r') as file:
    return json.load(file)

def save_json(path: Path, data: Any) -> None:
  with path.open(mode='w') as file:
    json.dump(data, file, indent=2)

def load_str(path: Path) -> str:
  with path.open(mode='r') as file:
    return ''.join(file.readlines()).strip().rstrip('\n')

def save_str(path: Path, data: str) -> str:
  with path.open(mode='w') as file:
    file.write(data)

def readline_completer(options: list[str]):
  def completer(text: str, state: int):
    matches = [option for option in options if option.startswith(text)]
    if state < len(matches):
      return matches[state]
    else:
      return None
  return completer

def input_option(
    prompt: str,
    options: list[str],
    default: str|int = 0,
    required: bool = False,
    blacklist: list[str] = [],
) -> str:
  if len(options) == 0:
    options.append('')
  options = list(map(str, options))
  readline.set_completer(readline_completer(options))
  default = options[default]
  while True:
    print(f'{prompt} [bold cyan]({default})[/bold cyan]: ')
    option = input().strip()
    if option == '':
      if required:
        continue
      option = default
    if option in blacklist:
      print(f'[bold red]{option} is not allowed.[/bold red]')
      continue
    readline.set_completer(readline_completer([]))
    return option

def walk_dir(dir: Path) -> list[Path]:
  paths = sorted(
      Path(dir).iterdir(),
      key=lambda path: (path.is_file(), path.name.lower()),
  )
  files = []
  for path in paths:
    if path.name.startswith("."):
      continue
    if path.is_dir():
      files.extend(walk_dir(path))
    else:
      files.append(path)
  return files

def walk_dir_relative(dir: Path) -> list[Path]:
  return [file.relative_to(dir) for file in walk_dir(dir)]