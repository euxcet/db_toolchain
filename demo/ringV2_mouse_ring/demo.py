import time
import argparse
from db_graph.framework.graph import Graph
from db_graph.utils.file_utils import load_json
from db_graph.utils.config import fill_value_by_name
from db_zoo.node.algorithm.dynamic_gesture_detector import DynamicGestureDetector
from db_zoo.node.algorithm.trajectory_detector import TrajectoryDetector
from db_zoo.node.device.ringV2 import RingV2
from mouse_ring import MouseRing

def run():
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, default='config/ring_config.json')
  parser.add_argument('-r', '--ring', type=str, default=None)
  parser.add_argument('-p', '--checkpoint', type=str, default=None)
  args = parser.parse_args()

  config = load_json(args.config)
  fill_value_by_name(config, 'Ring', 'address', args.ring)
  fill_value_by_name(config, 'TrajectoryDetector', 'checkpoint_file', args.checkpoint)

  Graph(config).run()

  while True:
    time.sleep(1)

if __name__ == '__main__':
  run()