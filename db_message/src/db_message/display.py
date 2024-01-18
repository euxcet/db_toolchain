from pathlib import Path
from rich.filesize import decimal
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.text import Text
from rich.markup import escape
from .repository import Repository
from .git_repository import GitRepository

def walk_directory(directory: Path, tree: Tree) -> None:
  paths = sorted(
    Path(directory).iterdir(),
    key=lambda path: (path.is_file(), path.name.lower()),
  )
  for path in paths:
    if path.name.startswith("."):
      continue
    if path.is_dir():
      style = "dim" if path.name.startswith("__") else ""
      branch = tree.add(
        f"[bold magenta]:open_file_folder: [link file://{path}]{escape(path.name)}",
        style=style,
        guide_style=style,
      )
      walk_directory(path, branch)
    else:
      text_filename = Text(path.name, "green")
      text_filename.highlight_regex(r"\..*$", "bold red")
      text_filename.stylize(f"link file://{path}")
      file_size = path.stat().st_size
      text_filename.append(f" ({decimal(file_size)})", "blue")
      icon = "👉 " if path.suffix == '.proto' else '📄 '
      tree.add(Text(icon) + text_filename)

def show_dir_tree(directory: Path):
  console = Console()
  tree = Tree(
      f":open_file_folder: [link file://{directory}]{directory}",
      guide_style="bold bright_blue",
  )
  walk_directory(directory, tree)
  console.print(tree)

def show_repos(repos: list[Repository]):
  console = Console()
  table = Table(show_header=True, header_style="bold blue")
  table.add_column("Name")
  table.add_column("Type")
  table.add_column("Local")
  table.add_column("Remote")
  for repo in repos:
    table.add_row(*repo.to_row())
  console.print(table)

def show_git_repo_info(repo: GitRepository):
  console = Console()
  table = Table(show_header=True, header_style="bold blue")
  table.add_column("Name")
  table.add_column("Branch")
  table.add_column("Commit Hex")
  table.add_column("Commit Author")
  table.add_column("Commit Message")
  table.add_row(*repo.info)
  console.print(table)