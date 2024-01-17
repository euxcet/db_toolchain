import time
import argparse
from .node import *
from .device import *
from .framework.node_manager import node_manager
from .utils.file_utils import load_json
from .utils.config import fill_value_by_type

def run():
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, required=True)
  parser.add_argument('-r', '--ring', type=str, default=None)
  parser.add_argument('-g', '--glove', type=str, default=None)
  args = parser.parse_args()

  config = load_json(args.config)
  fill_value_by_type(config, 'Ring', 'address', args.ring)
  fill_value_by_type(config, 'Glove', 'ip', args.glove)

  node_manager.add_nodes(config)

  while True:
    time.sleep(1)

if __name__ == '__main__':
  run()