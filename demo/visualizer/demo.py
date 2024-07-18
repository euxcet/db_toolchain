import time
import argparse
from db_graph.framework.graph import Graph
from db_graph.utils.file_utils import load_json
from db_graph.utils.config import fill_value_by_name
from db_zoo.node.mock.mock import Mock
from db_zoo.node.visualizer.visualizer import Visualizer

def run():
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, default='config/config.json')
  parser.add_argument('-r', '--ring', type=str, default=None)
  args = parser.parse_args()

  config = load_json(args.config)

  Graph(config).run()

  while True:
    time.sleep(1)

if __name__ == '__main__':
  run()
