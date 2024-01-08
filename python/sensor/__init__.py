from .device import Device, DeviceLifeCircleEvent
from .ring import Ring, RingConfig, RingAction
from .ring_pool import ring_pool
from .glove import Glove, GloveConfig, GloveEvent, GloveEventType, glove_pool
from .data.glove_data import GloveIMUJointName, GloveQuaternionJointName

def add_device(config:dict, ring_address:str=None, glove_ip:str=None):
  if config['type'] == 'ring':
    if ring_address is not None:
      config['address'] = ring_address
    return ring_pool.add_device(RingConfig(**config))
  elif config['type'] == 'glove':
    if glove_ip is not None:
      config['ip'] = glove_ip
    return glove_pool.add_device(GloveConfig(**config))
  raise Exception(f"Unknown device type {config['type']}")