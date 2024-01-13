import time
from sensor.pool import Pool
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from sensor.device import DeviceLifeCircleEvent
from sensor.ring import Ring, RingConfig
from utils.logger import logger

class RingPool(Pool):
  def __init__(self, keep_alive=True):
    self.keep_alive = keep_alive
    # key is the address of the ring
    self.rings:dict[str, Ring] = {}
    self.executor = ThreadPoolExecutor(max_workers=20, thread_name_prefix='Ring')
    if self.keep_alive:
      keep_alive_thread = Thread(target=self.keep_alive_strategy)
      keep_alive_thread.daemon = True
      keep_alive_thread.start()

  def keep_alive_strategy(self): ...

  def add_device(self, config:RingConfig, wait_until_connect:bool=False) -> Ring:
    if config.address in self.rings:
      logger.warning(f'Ring[{config.address}] is already added.')
      return
    ring = Ring(config)
    self.rings[config.address] = ring
    self.executor.submit(ring.connect_sync)
    if wait_until_connect:
      while ring.lifecycle_status != DeviceLifeCircleEvent.on_connect:
        time.sleep(0.3)
    return ring

  def get_device(self, config:RingConfig, connect_when_miss:bool=None) -> Ring:
    if config.address in self.rings:
      return self.rings[config.address]
    if connect_when_miss:
      return self.add_device(config)
    logger.warning(f'The required ring[{config.address}] has not been added yet, \
                     please consider using the connect_when_miss parameter.')
    return None

ring_pool = RingPool()