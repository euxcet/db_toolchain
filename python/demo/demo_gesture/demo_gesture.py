import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../..'))
import time
import argparse
from utils.file_utils import load_json
from sensor import add_device
from demo.detectors import *
from demo.demo_gesture.gesture_aggregator import GestureAggregator # DO NOT REMOVE THIS LINE
from demo.detector import detector_register

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, default='config/ring_only_config.json')
  parser.add_argument('-r', '--ring', type=str, default=None)
  parser.add_argument('-g', '--glove', type=str, default=None)
  args = parser.parse_args()

  config = load_json(args.config)
  devices = {
    cfg['name']: add_device(cfg, ring_address=args.ring, glove_ip=args.glove)
    for cfg in config['devices']
  }
  detectors = {
    cfg['name']: detector_register.instance(cfg['type'], ignore_keys=['type'], kwargs=cfg)
    for cfg in config['detectors']
  }
  while True:
    time.sleep(1)

# "labels": ["negative", "g_hostage", "g_gather", "g_quick", "g_stop", "g_search", "g_ok", "g_safe", "g_danger", "g_prepare", "g_sniper", "thumbright", "thumbleft", "handforwardhorizontal", "handforwardvertical", "pinkyout", "aim", "fingergatherup", "fingergatherdown", "indexout", "indexdown", "palmout", "palmin", "rest", "fingergatherright"],