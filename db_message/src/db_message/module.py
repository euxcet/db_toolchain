from __future__ import annotations

import shutil
import readline
from rich import print, print_json
from rich.text import Text
from rich.markdown import Markdown
from pathlib import Path
from .repository_manager import repository_manager
from .git_repository import GitRepository
from .display import show_dir_tree
from .utils import load_json, save_json
from .utils import load_str, save_str
from .utils import input_option
from .compiler import Compiler, compiler_register
from .compilers import *

class Module():
  DB_MESSAGE_CONFIG = 'dbm_module.json'

  def __init__(self, path: Path):
    self.path = path

  def add(self) -> None:
    config: dict = self.load_config()
    # Only support GitRepository right now.
    print(Text('Add a Synchronous File to this Module', style="green"))
    repo:GitRepository = repository_manager.get(input_option(
      prompt='Repository',
      options=repository_manager.names(), 
    ))

    status = repo.store()

    head = input_option(prompt='Head', options=repo.head_names())
    commit = input_option( prompt='Commit', options=repo.commit_hexes(head))
    show_dir_tree(repo.local)
    file = input_option(prompt='File', options=repo.files(commit))
    compiler = input_option(prompt='Compiler', options=['TextCompiler', 'ProtoCompiler'])
    save_filename = input_option(prompt='Save filename', options=[''], required=True,
                             blacklist=list(map(lambda x:x['save_filename'], config['files'])))

    file_config = {
      'repository': repo.name,
      'branch': head,
      'commit': commit,
      'file': file,
      'compiler': compiler,
      'save_filename': save_filename,
    }
    config['files'].append(file_config)
    self.save_config(config)

    shutil.copy(Path(repo.local, file), Path(save_filename))
    compiler_instance:Compiler = compiler_register.instance(compiler)
    compiler_instance.compile(Path(save_filename))

    repo.restore(status)

  def update_file(self, file_config: dict) -> None:
    repo:GitRepository = repository_manager.get(file_config['repository'])
    status = repo.store()
    repo.checkout(file_config['branch'])
    repo.checkout(file_config['commit'])
    shutil.copy(Path(repo.local, file_config['file']), Path(file_config['save_filename']))
    compiler_instance:Compiler = compiler_register.instance(file_config['compiler'])
    compiler_instance.compile(Path(file_config['save_filename']))
    repo.restore(status)

  def update(self, save_filename: str = None) -> None:
    config: dict = self.load_config()
    for file_config in config['files']:
      # TODO wildcard matching
      if save_filename is None or file_config['save_filename'] == save_filename:
        self.update_file(file_config)

  def remove(self, filename: str) -> None:
    config:dict = self.load_config()
    for file_config in config['files']:
      # TODO wildcard matching
      if file_config['save_filename'] == filename:
        config['files'].remove(file_config)
        print(Text(f'{filename} removed.', style="green"))
        self.save_config(config)
        return
    print(Text(f'{filename} not found.', style="red"))

  def load_config(self) -> dict:
    return load_json(Path(self.path, self.DB_MESSAGE_CONFIG))

  def save_config(self, config: dict):
    save_json(Path(self.DB_MESSAGE_CONFIG), config)

  def create(path: Path) -> Module:
    name = input('Module name: ').strip()
    module_path = Path.cwd() / name
    if module_path.exists():
      raise FileExistsError(module_path)
    module_path.mkdir()
    save_str(Path(module_path, '.gitignore'), f'*\n!{Module.DB_MESSAGE_CONFIG}')
    save_json(Path(module_path, Module.DB_MESSAGE_CONFIG), {'files': []})
    return Module(module_path)
