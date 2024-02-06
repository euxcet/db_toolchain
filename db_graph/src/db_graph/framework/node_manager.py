from .node import Node, node_register
from .device import Device
from .device_manager import DeviceManager
from .edge_manager import EdgeManager
from ..utils.config import VALID_KEYS

class NodeManager():
  def __init__(self, edge_manager: EdgeManager):
    self.nodes: dict[str, Node] = {}
    self.edge_manager = edge_manager
    self.device_manager = DeviceManager()

  def add_nodes(self, config: dict):
    for key in VALID_KEYS:
      if key in config:
        for node_config in config[key]:
          node_config['edge_manager'] = self.edge_manager
          node = node_register.instance(node_config['type'], ignore_keys=['type'], kwargs=node_config)
          self.nodes[node_config['name']] = node
          if issubclass(type(node), Device):
            self.device_manager.add_device(node)

  def start(self):
    self.device_manager.start()

node_manager = NodeManager()