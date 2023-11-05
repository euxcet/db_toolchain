import queue
import asyncio
from threading import Thread
from .ring_ble import RingBLE, RingEvent, RingConfig
from concurrent.futures import ThreadPoolExecutor
from utils.logger import logger

class Ring():
  def __init__(self, config:RingConfig, event_queue:queue.Queue):
    self.config = config
    self.address = config.address
    self.event_queue = event_queue
    self.ring = RingBLE(config, event_callback=self.event_callback)
    self.disconnected = False
    self.imu_data_queue = queue.Queue()

  async def connect_rings(self):
    coroutines = [ring.connect() for ring in self.rings]
    await asyncio.gather(*coroutines)

  def event_callback(self, event:RingEvent):
    self.event_queue.put_nowait(event)

  def blink(self, blink_color, blink_time):
    # TODO: use asyncio?
    Thread(target=self.ring.blink(blink_color, blink_time)).start()

  def set_color(self, color):
    self.ring.set_color(color)

  def send_action(self, action):
    self.ring.send_action(action)

  def run(self):
    asyncio.run(self.connect_rings())

class RingPool():
  def __init__(self):
    # key is the address of the ring
    self.rings:dict[str, Ring] = {}
    self.handlers:dict[str, set] = {}
    self.executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix='Ring')
    self.event_queue:queue.Queue[RingEvent] = queue.Queue()
    event_handler_thread = Thread(target=self.event_handler)
    event_handler_thread.daemon = True
    event_handler_thread.start()

  # TODO: use config to store address, ...
  def add_ring(self, config:RingConfig) -> Ring:
    if config.address in self.rings:
      logger.warning(f'Ring[{config.address}] is already added.')
      return
    ring = Ring(config, event_queue=self.event_queue)
    self.rings[config.address] = ring
    self.handlers[config.address] = set()
    self.executor.submit(ring.run)

  def get_ring(self, config:RingConfig, connect_when_miss:bool=None) -> Ring:
    if config.address in self.rings:
      return self.rings[config.address]
    if connect_when_miss:
      return self.add_ring(config)
    logger.warning(f'The required ring[{config.address}] has not been added yet, \
                     please consider using the connect_when_miss parameter.')
    return None

  def bind_ring(self, event_handler, ring:Ring=None, config:RingConfig=None, address:str=None):
    if ring is not None:
      ring_address = ring.address
    elif config is not None:
      ring_address = config.address
    elif address is not None:
      ring_address = address
    else:
      ring_address = None
    if ring_address is None:
      logger.error(f'The address for the ring is not provided.')
      raise Exception()
    self.handlers[ring_address].add(event_handler)

  def event_handler(self):
    while True:
      event = self.event_queue.get()
      for handler in self.handlers[event.address]:
        handler(self.rings[event.address], event)

ring_pool = RingPool()