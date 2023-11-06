import time
import torch
import socket
import struct
import argparse
import numpy as np
import torch.nn.functional as F
from core.imu_data import IMUData
from python.model.gesture_cnn import GestureNetCNN
from python.utils.window import Window
from threading import Thread
from python.model.fully_connected import FullyConnectedModel
from queue import Queue
from python.core.ring_ble import RingBLE, scan_rings

class GloveGestureDetector(Thread):
  def __init__(self, handler, out_ip:str=None):
    Thread.__init__(self)
    self.counter = Counter()
    self.model = FullyConnectedModel(class_num=11)
    self.model.load_state_dict(torch.load('/home/euxcet/gesture_train/glove.pth'))
    self.model.eval()
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.sock.connect(('127.0.0.1', 11002))
    self.handler = handler
    self.out_sock = None

    self.gesture_name = {
      0: 'sniper',
      1: 'hostage',
      2: 'gather',
      3: 'quick',
      4: 'stop',
      5: 'search',
      6: 'ok',
      7: 'safe',
      8: 'danger',
      9: 'prepare',
      # 10: 'forward'
    }
    
    print('Glove connected.')

    if out_ip is not None:
      server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      server_sock.bind((out_ip, 11002))
      server_sock.listen(2)
      self.out_sock, _ = server_sock.accept()
      print('Glove output socket connected.')

  def run(self):
    gesture_window = Window(6)
    last_gesture_time = 0
    while True:
      data = self.sock.recv(1024)
      if self.out_sock is not None:
        self.out_sock.send(data)
      if data.decode('cp437').find('VRTRIX') == 0:
        radioStrength, = struct.unpack('h', data[265:267])
        battery, = struct.unpack('f', data[267:271])
        calScore, = struct.unpack('h', data[271:273])
        joint_quat = [struct.unpack('f', data[9 + 4 * i: 9 + 4 * i + 4])[0] for i in range(64)]
        self.counter.count({'RadioStrength': -radioStrength, 'Battery': battery, 'CalScore': calScore})
        if self.counter.counter % 10 == 0:
          input_tensor = torch.tensor(np.array(joint_quat).astype(np.float32).reshape(1, 64))
          outputs = F.softmax(self.model(input_tensor).cpu().detach(), dim=1)
          _, predictions = torch.max(outputs, 1)
          gesture, confidence = predictions[0].item(), outputs[0][predictions[0].item()].item()
          if gesture in self.gesture_name and confidence > 0.95:
            gesture_window.push(gesture)
            current_time = time.time()
            if current_time > last_gesture_time + 1.2 and gesture_window.full() and gesture_window.count(lambda x:x == gesture_window.first()) == gesture_window.capacity():
              self.handler(self.gesture_name[gesture])
              last_gesture_time = current_time


class GestureDetector():
  def __init__(self, args):
    ring_data_queue = Queue()
    ring = Ring(args.ring, ring_data_queue, args.adapter)
    ring_detector = RingGestureDetector(self.ring_gesture_handler, ring_data_queue)
    glove_detector = GloveGestureDetector(self.glove_gesture_handler)

    ring.setDaemon(True)
    ring_detector.setDaemon(True)
    glove_detector.setDaemon(True)

    ring.start()
    ring_detector.start()
    glove_detector.start()
    count = 0
    while True:
      if count > 15 and not ring.connected:
        break
      count += 1
      time.sleep(1)

  def glove_gesture_handler(self, glove_gesture):
    print('Glove', glove_gesture)

  def ring_gesture_handler(self, ring_gesture):
    print('Ring', ring_gesture)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--devices', type=str)
  parser.add_argument('--ring', type=str)
  parser.add_argument('--glove_ip', type=str, default='127.0.0.1')
  parser.add_argument('--glove_port', type=int, default=11002)
  parser.add_argument('--adapter', type=str, default='hci0')
  args = parser.parse_args()
  GestureDetector(args)
