import time
import argparse
from db_core.node import *
from db_core.device import *
from db_core.framework.node_manager import node_manager
from db_core.utils.file_utils import load_json
from db_core.utils.config import fill_value_by_name
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

  node_manager.add_nodes(config)
  node_manager.start()

  while True:
    time.sleep(1)

if __name__ == '__main__':
  run()