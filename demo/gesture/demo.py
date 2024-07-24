import time
import argparse
from db_graph.framework.graph import Graph
from db_graph.utils.file_utils import load_json
from db_graph.utils.config import fill_value_by_name
from gesture_aggregator import GestureAggregator
from db_zoo.node.algorithm.dynamic_gesture_detector import DynamicGestureDetector
from db_zoo.node.algorithm.static_gesture_detector import StaticGestureDetector
from db_zoo.node.device.ring import Ring

def run():
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, default='config/config.json')
  parser.add_argument('-r', '--ring', type=str, default=None)
  parser.add_argument('-g', '--glove', type=str, default=None)
  args = parser.parse_args()

  config = load_json(args.config)
  fill_value_by_name(config, 'Ring', 'address', args.ring)
  fill_value_by_name(config, 'Glove', 'ip', args.glove)

  Graph(config).run()

  while True:
    time.sleep(1)

if __name__ == '__main__':
  run()