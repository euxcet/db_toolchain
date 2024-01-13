# from .ring import Ring
# from .glove import Glove
# from .glove_imu import GloveIMU
# from .timestamp import Timestamp
from utils.logger import logger

def get_devices(config):
  print(config)
  for device in config:
    print(device)
  return []
  # device_dict = {
  #   'ring': Ring(),
  #   'glove': Glove(),
  #   'glove_imu': GloveIMU(),
  #   'timestamp': Timestamp(),
  # }
  # devices_name = set([x.lower() for x in device_str.strip().split(',')])
  # devices = []
  # for name in devices_name:
  #   if name in device_dict:
  #     devices.append(device_dict[name])
  #   else:
  #     logger.error(f"Device [{name}] not found")
  #     raise KeyError()
  # return devices