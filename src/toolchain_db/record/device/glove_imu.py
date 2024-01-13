import time
import socket
import struct
import numpy as np
import os.path as osp
from utils.counter import Counter
from utils.file_utils import make_dir
from threading import Thread

class GloveIMU(Thread):
  def __init__(self, ip:str='169.254.6.2'):
    Thread.__init__(self)
    self.counter = Counter()
    self.save_file = None
    self.running = True
    self.recording = False
    self.save_dir = None

  def process_data(self, data):
    if data.decode('cp437').find('VRTRIX') == 0:
      radioStrength, = struct.unpack('h', data[317:319])
      battery, = struct.unpack('f', data[319:323])
      calScore, = struct.unpack('h', data[323:325])
      self.counter.count({'RadioStrength': -radioStrength, 'Battery': battery, 'CalScore': calScore})
      raw_data = [struct.unpack('f', data[9 + 4 * i: 9 + 4 * i + 4])[0] for i in range(77)]
      if self.recording:
        self.save_file.write(struct.pack('i', 0))
        for i in range(77):
          self.save_file.write(struct.pack('f', raw_data[i]))
        self.save_file.write(struct.pack('d', time.time()))

  def run(self):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (self.ip, 11002)
    print('[Glove] Connecting to %s port %s' % server_address)
    sock.connect(server_address)
    print('[Glove] Connected.')

    while self.running:
      data = sock.recv(1024)
      self.process_data(data)

  def get_save_file(self, round_id):
    return osp.join(self.save_dir, str(round_id) + '_glove.bin')

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
    self.flush()
    make_dir(self.save_dir)
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
