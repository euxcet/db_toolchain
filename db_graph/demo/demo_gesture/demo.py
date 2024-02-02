import time
import argparse
from db_graph.node import *
from db_graph.device import *
from db_graph.framework.node_manager import node_manager
from db_graph.utils.file_utils import load_json
from db_graph.utils.config import fill_value_by_name
from gesture_aggregator import GestureAggregator

def run():
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, default='config/ring_only_config.json')
  parser.add_argument('-r', '--ring', type=str, default=None)
  parser.add_argument('-g', '--glove', type=str, default=None)
  args = parser.parse_args()

  config = load_json(args.config)
  fill_value_by_name(config, 'Ring', 'address', args.ring)
  fill_value_by_name(config, 'Glove', 'ip', args.glove)

  node_manager.add_nodes(config)
  node_manager.start()

  while True:
    time.sleep(1)

if __name__ == '__main__':
  run()