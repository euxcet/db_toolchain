import time
import queue
import asyncio
from threading import Thread
from core.ble_ring import BLERing, RingEvent, RingEventType, RingLifeCircleEvent
from core.imu_data import IMUData
from concurrent.futures import ThreadPoolExecutor

class Ring():
  # TODO: use config to store address, ...
  def __init__(self, address:str, name:str, event_queue:queue.Queue, adapter:str=None):
    self.address = address
    self.name = name
    self.event_queue = event_queue
    self.ring = BLERing(address=address, name=name, event_callback=self.event_callback, adapter=adapter)
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
    self.rings = {} # address: ring
    # create a thread for each ring
    self.executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix='Ring')
    self.event_queue = queue.Queue()
    event_handler_thread = Thread(target=self.event_handler)
    event_handler_thread.daemon = True
    event_handler_thread.start()

  # TODO: use config to store address, ...
  def add_ring(self, address, name:str='Ring Unnamed', adapter:str=None):
    if address in self.rings:
      raise Exception()
    ring = Ring(address, name=name, event_queue=self.event_queue, adapter=adapter)
    self.rings[address] = ring
    self.executor.submit(ring.run)

  def get_ring(self, address, connect_when_miss:bool=None, name:str=None, adapter:str=None):
    pass

  def event_handler_thread(self):
    pass

ring_pool = RingPool()