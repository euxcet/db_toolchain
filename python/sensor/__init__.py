from .ring import Ring, ring_pool
from .ring_ble import RingConfig, RingEvent, RingEventType, RingLifeCircleEvent

from .glove import Glove, GloveConfig, GloveEvent, GloveEventType, glove_pool
from .glove_data import GloveIMUJointName, GloveQuaternionJointName

def add_device(config:dict, ring_address:str=None, glove_ip:str=None):
  if config['type'] == 'ring':
    if ring_address is not None:
      config['address'] = ring_address
    return ring_pool.add_ring(RingConfig(**config))
  elif config['type'] == 'glove':
    if glove_ip is not None:
      config['ip'] = glove_ip
    return glove_pool.add_glove(GloveConfig(**config))
  raise Exception(f"Unknown device type {config['type']}")