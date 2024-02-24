from concurrent.futures import ThreadPoolExecutor
from .node_manager import NodeManager
from .edge_manager import EdgeManager

class Graph():
  def __init__(self, config: dict):
    self.edge_manager = EdgeManager()
    self.node_manager = NodeManager(self)
    self.node_manager.add_nodes(config)

  def run(self):
    self.node_manager.start()

  @property
  def executor(self) -> ThreadPoolExecutor:
    return self.node_manager.executor
