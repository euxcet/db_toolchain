from pathlib import Path
from xdg_base_dirs import xdg_data_home
from .repository import Repository
from .git_repository import GitRepository

class RepositoryManager():
  def __init__(self, save_dir: Path):
    self.save_dir = save_dir
    self.repos: list[Repository] = self.load_from_dir(self.save_dir)

  def load_from_dir(self, save_path: Path) -> list[Repository]:
    if not save_path.exists():
      save_path.mkdir()
    repos = []
    for repo_path in save_path.iterdir():
      if repo_path.is_dir():
        for repo_class in [GitRepository]:
          if repo_class.check(repo_path):
            repos.append(repo_class(repo_path))
    return repos

  def clone(self, url: str) -> Repository:
    if url.endswith('.git'):
      repo = GitRepository.clone(url, self.save_dir / url.split('/')[-1][0:-4])
    return repo

  def get(self, name: str) -> Repository:
    for repo in self.repos:
      if repo.name == name:
        return repo
    raise KeyError(f'Repository <name> does not exist.')

  def update(self, name: str) -> Repository:
    return self.get(name).update()

  def names(self) -> list[str]:
    return list(map(lambda x:x.name, self.repos))

repository_manager = RepositoryManager(xdg_data_home() / 'dbm')