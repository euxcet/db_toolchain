from concurrent.futures import ThreadPoolExecutor
from .node import Node, node_register
from ..utils.config import VALID_KEYS

class NodeManager():
  def __init__(self, graph):
    from .graph import Graph
    self.graph: Graph = graph
    self.nodes: dict[str, Node] = {}
    self.executor = ThreadPoolExecutor(max_workers=20, thread_name_prefix='Node')

  def add_nodes(self, config: dict):
    for key in VALID_KEYS:
      if key in config:
        for node_config in config[key]:
          node_config['graph'] = self.graph
          node = node_register.instance(node_config['type'], ignore_keys=['type'], kwargs=node_config)
          self.nodes[node_config['name']] = node

  def start(self):
    for node in self.nodes.values():
      node.start()

  def block(self):
    for node in self.nodes.values():
      node.block()

