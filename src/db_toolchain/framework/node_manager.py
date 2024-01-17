from .node import Node, node_register
from .device import Device
from .device_manager import device_manager
from ..utils.config import VALID_KEYS

class NodeManager():
  def __init__(self):
    self.nodes: dict[str, Node] = {}

  def add_nodes(self, config: dict):
    for key in VALID_KEYS:
      if key in config:
        for node_config in config[key]:
          node = node_register.instance(node_config['type'], ignore_keys=['type'], kwargs=node_config)
          self.nodes[node_config['name']] = node
          if issubclass(type(node), Device):
            device_manager.add_device(node)

node_manager = NodeManager()