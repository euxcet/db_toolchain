from __future__ import annotations

import git
from git import Repo
from pathlib import Path
from .repository import Repository
from .utils import walk_dir_relative

class GitRepositoryStatus():
  def __init__(self, branch: str, commit: str):
    self.branch = branch
    self.commit = commit

class GitRepository(Repository):
  def __init__(self, local: Path):
    self.repo = Repo(local)
    super(GitRepository, self).__init__(
      name=local.name,
      local=local,
      remote=self.repo.remotes[0].url,
    )

  @property
  def info(self) -> list[str]:
    return [
      self.name,
      self.repo.head.name,
      self.repo.head.commit.hexsha,
      self.repo.head.commit.author.name,
      self.repo.head.commit.message,
    ]

  def progress_callback(op_code, cur_count, max_count=None, message=''):
    # TODO: use rich.Progress
    if op_code == git.RemoteProgress.COMPRESSING:
        print(f"Compressing: {cur_count}/{max_count} - {message}")
    elif op_code == git.RemoteProgress.RECEIVING:
        print(f"Receiving: {cur_count}/{max_count} - {message}")
    elif op_code == git.RemoteProgress.RESOLVING:
        print(f"Resolving: {cur_count}/{max_count} - {message}")
    elif op_code == git.RemoteProgress.COUNTING:
        print(f"Counting: {cur_count}/{max_count} - {message}")

  def clone(url: str, local: Path) -> GitRepository:
    Repo.clone_from(url, local, progress=GitRepository.progress_callback)
    return GitRepository(local)

  def update(self) -> GitRepository:
    remote = self.repo.remote()
    remote.pull()
    self.repo.submodule_update()
    return self

  def check(path: Path) -> bool:
    try:
      Repo(path)
      return True
    except:
      return False

  def checkout(self, commit) -> None:
    self.repo.git.checkout(commit)

  def head_names(self) -> list[str]:
    heads = []
    try:
      heads.append(self.repo.active_branch)
    except: ...
    for head in self.repo.heads:
      if head not in heads:
        heads.append(head)
    return list(map(lambda x:x.name, heads))

  def commit_hexes(self, branch: str) -> list[str]:
    self.checkout(branch)
    return [commit.name_rev.split(' ')[0] for commit in self.repo.iter_commits()]

  def files(self, commit: str) -> list[str]:
    self.checkout(commit)
    return walk_dir_relative(self.local)

  def store(self) -> GitRepositoryStatus:
    try:
      branch = self.repo.active_branch.name
    except:
      head_names = self.head_names()
      for b in ['main', 'master', 'dev', 'develop', head_names[0]]:
        if b in head_names:
          branch = b
    return GitRepositoryStatus(
      branch=branch,
      commit=self.repo.head.commit.name_rev,
    )
  
  def restore(self, status: GitRepositoryStatus) -> None:
    self.repo.git.checkout(status.branch)