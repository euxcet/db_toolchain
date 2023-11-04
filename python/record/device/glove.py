import socket
import struct
import numpy as np
import os.path as osp
from utils.counter import Counter
from utils.file_utils import make_dir
from threading import Thread

class Glove(Thread):
  def __init__(self, ip:str='169.254.6.2'):
    Thread.__init__(self)
    self.counter = Counter()
    self.save_file = None
    self.running = True
    self.recording = False
    self.save_dir = None
    self.data = []

  def process_data(self, data):
    if data.decode('cp437').find('VRTRIX') == 0:
      radioStrength, = struct.unpack('h', data[265:267])
      battery, = struct.unpack('f', data[267:271])
      calScore, = struct.unpack('h', data[271:273])
      self.counter.count({'RadioStrength': -radioStrength, 'Battery': battery, 'CalScore': calScore})
      if self.running:
        if self.counter.counter % 10 == 0:
          self.data.append(np.array([struct.unpack('f', data[9 + 4 * i: 13 + 4 * i])[0] for i in range(64)]).reshape(1, 64))

  def run(self):
    print('[Glove] Connecting to %s port %s' % server_address)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (self.ip, 11002)
    sock.connect(server_address)
    print('[Glove] Connected.')

    while self.running:
      data = sock.recv(1024)
      self.process_data(data)

  def get_save_file(self, round_id):
    return osp.join(self.save_dir, str(round_id) + '_glove.npy')

  # override
  def set_save_dir(self, save_dir):
    self.save_dir = save_dir

  # override
  def stop(self):
    self.running = False

  # override
  def begin_round(self, round_id):
    make_dir(self.save_dir)
    self.recording = True
    self.data = []

  # override
  def end_round(self, round_id):
    np.save(self.get_save_file(round_id), np.concatenate(self.data))
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
