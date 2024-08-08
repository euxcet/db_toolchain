import time
import argparse
from db_graph.framework.graph import Graph
from db_graph.utils.file_utils import load_config
from db_graph.utils.config import fill_value_by_name
from db_zoo.node.algorithm.dynamic_gesture_detector import DynamicGestureDetector
from db_zoo.node.algorithm.static_gesture_detector import StaticGestureDetector
from db_zoo.node.algorithm.gesture_aggregator import GestureAggregator
from db_zoo.node.device.ring import Ring
from db_zoo.node.device.tello import Tello
from db_zoo.node.device.gshx_ar import GshxAR
from drone_controller import DroneController

def run():
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, default='config/config.yaml')
  parser.add_argument('-i', '--ip', type=str, default=None)
  parser.add_argument('-a', '--arip', type=str, default=None)
  args = parser.parse_args()

  config = load_config(args.config)
  fill_value_by_name(config, 'Tello', 'ip', args.ip)
  fill_value_by_name(config, 'Tello', 'ar_video_ip', args.arip)
  fill_value_by_name(config, 'GshxAR', 'ip', args.arip)

  Graph(config).run()

  while True:
    time.sleep(1)

if __name__ == '__main__':
  run()