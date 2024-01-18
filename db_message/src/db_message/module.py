from __future__ import annotations

import shutil
import readline
from rich import print
from rich.text import Text
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
    config:dict = load_json(Path('./dbm_module.json'))
    print(config)
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
    save_json(Path('./dbm_module.json'), config)

    shutil.copy(Path(repo.local, file), Path(save_filename))
    compiler_instance:Compiler = compiler_register.instance(compiler)
    compiler_instance.compile(Path(save_filename))

    repo.restore(status)

  def update(self) -> None:
    config:dict = load_json(Path('./dbm_module.json'))
    for file_config in config['files']:
      repo:GitRepository = repository_manager.get(file_config['repository'])
      status = repo.store()
      repo.checkout(file_config['branch'])
      repo.checkout(file_config['commit'])
      shutil.copy(Path(repo.local, file_config['file']), Path(file_config['save_filename']))
      compiler_instance:Compiler = compiler_register.instance(file_config['compiler'])
      compiler_instance.compile(Path(file_config['save_filename']))
      repo.restore(status)

  def create(path: Path) -> Module:
    name = input('Module name: ').strip()
    module_path = Path.cwd() / name
    if module_path.exists():
      raise FileExistsError(module_path)
    module_path.mkdir()
    save_str(Path(module_path, '.gitignore'), f'*\n!{Module.DB_MESSAGE_CONFIG}')
    save_json(Path(module_path, Module.DB_MESSAGE_CONFIG), {'files': []})
    return Module(module_path)
