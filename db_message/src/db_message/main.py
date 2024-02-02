import click
import readline
from pathlib import Path
from rich import print_json
from .display import show_repos, show_git_repo_info, show_dir_tree

readline.parse_and_bind('set editing-mode vi')
readline.parse_and_bind('tab: complete')

@click.command()
def repo_list():
  from .repository_manager import repository_manager
  show_repos(repository_manager.repos)

@click.command()
@click.argument('name')
def repo_update(name: str):
  from .repository_manager import repository_manager
  repo = repository_manager.update(name)
  show_git_repo_info(repo)

@click.command()
@click.argument('name')
def repo_info(name: str):
  from .repository_manager import repository_manager
  repo = repository_manager.get(name)
  show_dir_tree(repo.local)
  show_git_repo_info(repo)

@click.command()
@click.argument('url')
def repo_clone(url: str):
  from .repository_manager import repository_manager
  repo = repository_manager.clone(url.strip())
  show_dir_tree(repo.local)
  show_repos([repo])

@click.group()
def repo():
  ...

@click.command()
@click.option('--path', required=False)
def add(path: str):
  from .module import Module
  module = Module(Path.cwd() if path is None else Path(path))
  module.add()

@click.command()
@click.argument('filename')
@click.option('--path', required=False)
def remove(filename: str, path: str):
  from .module import Module
  module = Module(Path.cwd() if path is None else Path(path))
  module.remove(filename)

@click.command()
@click.option('--name', required=False)
@click.option('--path', required=False)
def update(name: str, path: str):
  from .module import Module
  module = Module(Path.cwd() if path is None else Path(path))
  module.update(name)

@click.command()
@click.option('--path', required=False)
def config(path: str):
  from .module import Module
  module = Module(Path.cwd() if path is None else Path(path))
  print_json(data=module.load_config())

@click.command()
@click.option('--path', required=False)
def create(path: str):
  from .module import Module
  Module.create(Path.cwd() if path is None else Path(path))

@click.group()
def cli():
  ...


repo.add_command(repo_list, 'list')
repo.add_command(repo_update, 'update')
repo.add_command(repo_clone, 'clone')
repo.add_command(repo_info, 'info')
cli.add_command(repo)
cli.add_command(update)
cli.add_command(create)
cli.add_command(remove)
cli.add_command(config)
cli.add_command(add)

if __name__ == '__main__':
  cli()