import time
import queue
import asyncio
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from sensor.device import DeviceLifeCircleEvent
from sensor.ring import Ring, RingEvent, RingConfig
from utils.logger import logger

class RingPool():
  def __init__(self, keep_alive=True):
    self.keep_alive = keep_alive
    # key is the address of the ring
    self.rings:dict[str, Ring] = {}
    self.handlers:dict[str, set] = {}
    self.executor = ThreadPoolExecutor(max_workers=20, thread_name_prefix='Ring')
    self.event_queue:queue.Queue[RingEvent] = queue.Queue()
    event_distribution_thread = Thread(target=self.event_handler)
    event_distribution_thread.daemon = True
    event_distribution_thread.start()
    if self.keep_alive:
      keep_alive_thread = Thread(target=self.keep_alive_strategy)
      keep_alive_thread.daemon = True
      keep_alive_thread.start()

  def keep_alive_strategy(self):
    pass

  def add_ring(self, config:RingConfig, wait_until_connect:bool=False) -> Ring:
    if config.address in self.rings:
      logger.warning(f'Ring[{config.address}] is already added.')
      return
    ring = Ring(config, event_queue=self.event_queue)
    self.rings[config.address] = ring
    self.handlers[config.address] = set()
    self.executor.submit(ring.connect_sync)
    if wait_until_connect:
      while ring.lifecycle_status != DeviceLifeCircleEvent.on_connect:
        time.sleep(0.3)
    return ring

  def get_ring(self, config:RingConfig, connect_when_miss:bool=None) -> Ring:
    if config.address in self.rings:
      return self.rings[config.address]
    if connect_when_miss:
      return self.add_ring(config)
    logger.warning(f'The required ring[{config.address}] has not been added yet, \
                     please consider using the connect_when_miss parameter.')
    return None

  def bind_ring(self, event_handler, ring:Ring=None, config:RingConfig=None, address:str=None):
    ring_address = None if address is None else address
    ring_address = ring_address if config is None else config.address
    ring_address = ring_address if ring is None else ring.address
    if ring_address is None:
      logger.error(f'The address for the ring is not provided.')
      raise Exception()
    if ring_address not in self.rings:
      logger.error(f'Ring[{ring_address}] is not in the ring set.')
      raise Exception()
    self.handlers[ring_address].add(event_handler)

  def event_handler(self):
    while True:
      event = self.event_queue.get()
      for handler in self.handlers[event.address]:
        handler(self.rings[event.address], event)

ring_pool = RingPool(keep_alive=True)