import time
import struct
import asyncio
import os.path as osp
from core.ring_ble import RingBLE
from core.imu_data import IMUData
from utils.counter import Counter
from utils.file_utils import make_dir
from threading import Thread

class Ring(Thread):
  def __init__(self, mac:str='043C2EAC-6125-DEA0-7011-CF8D0D7DBEBA'):
    Thread.__init__(self)
    self.save_file = None
    self.all_data = bytearray()
    self.running = True
    self.recording = False
    self.save_dir = None
    self.counter = Counter()

    self.ring = RingBLE(mac, index=0, imu_callback=self.imu_callback)
    self.ring_thread = Thread(target=self.connect)
    self.ring_thread.daemon = True
    self.ring_thread.start()

  def connect(self):
    asyncio.run(self.ring.connect())

  # async
  def imu_callback(self, index:int, data:IMUData):
    if self.recording:
      self.save_file.write(struct.pack('i', index))
      for i in range(6):
          self.save_file.write(struct.pack('f', data[i]))
      self.save_file.write(struct.pack('d', time.time()))

  def get_save_file(self, round_id):
    return osp.join(self.save_dir, str(round_id) + '_ring.bin')

  def flush(self):
    if self.save_file is not None:
      self.save_file.flush()

  # override
  def set_save_dir(self, save_dir):
    self.save_dir = save_dir

  # override
  def stop(self):
    self.flush()
    self.running = False

  # override
  def begin_round(self, round_id):
    make_dir(self.save_dir)
    self.flush()
    self.save_file = open(self.get_save_file(round_id), 'wb')
    self.recording = True

  # override
  def end_round(self, round_id):
    self.flush()
    self.recording = False

  # override
  def begin_sample(self, sample_id):
    pass

  # override
  def end_sample(self, sample_id):
    pass

  # override
  def exist_save_file(self, round_id):
    return osp.exists(self.get_save_file(round_id))
