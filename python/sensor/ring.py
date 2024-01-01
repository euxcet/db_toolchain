import time
import queue
import asyncio
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from utils.logger import logger
from utils.counter import Counter
from .ring_ble import RingBLE, RingEvent, RingConfig, RingEventType, RingLifeCircleEvent

class Ring():
  def __init__(self, config:RingConfig, event_queue:queue.Queue):
    self.config = config
    self.address = config.address
    self.event_queue = event_queue
    self.ring = RingBLE(config, event_callback=self.event_callback)
    self.status = RingLifeCircleEvent.on_connecting
    self.imu_data_queue = queue.Queue()
    self.counter = Counter(print_interval=400)

  def event_callback(self, event:RingEvent):
    self.counter.count(enable_print=False)
    self.event_queue.put_nowait(event)
    if event.event_type == RingEventType.lifecircle:
      self.status = event.data

  def blink(self, blink_color, blink_time):
    # TODO: use asyncio
    Thread(target=self.ring.blink(blink_color, blink_time)).start()

  def set_color(self, color):
    self.ring.set_color(color)

  def send_action(self, action):
    self.ring.send_action(action)

  def run(self):
    asyncio.run(self.ring.connect())

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

  def add_ring(self, config:RingConfig, wait_until_initialized:bool=False) -> Ring:
    if config.address in self.rings:
      logger.warning(f'Ring[{config.address}] is already added.')
      return
    ring = Ring(config, event_queue=self.event_queue)
    self.rings[config.address] = ring
    self.handlers[config.address] = set()
    self.executor.submit(ring.run)
    if wait_until_initialized:
      while ring.status != RingLifeCircleEvent.on_initialized:
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