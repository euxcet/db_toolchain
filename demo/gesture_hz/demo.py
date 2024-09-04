import time
import argparse
from db_graph.framework.graph import Graph
from db_graph.utils.file_utils import load_config
from db_graph.utils.config import fill_value_by_name
from db_zoo.node.algorithm.gesture_aggregator import GestureAggregator
from db_zoo.node.device.ring import Ring
from db_zoo.node.device.tello import Tello
from db_zoo.node.device.gshx_ar import GshxAR
from gesture_detector import GestureDetector

def run():
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, default='config/config.yaml')
  args = parser.parse_args()

  config = load_config(args.config)
  Graph(config).run()

  while True:
    time.sleep(1)

if __name__ == '__main__':
  run()