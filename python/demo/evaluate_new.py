import os
import time
import math
import torch
import scipy
import socket
import struct
import argparse
import numpy as np
import torch.nn.functional as F
from core.imu_data import IMUData
from python.model.gesture_cnn import GestureNetCNN
from core.window import Window
from threading import Thread
from python.model.fully_connected import FullyConnectedModel
from queue import Queue
from core.ble_ring import BLERing, scan_rings
import playsound
import asyncio

class Counter():
  def __init__(self, print_gap:int=1000):
    self.t0 = 0
    self.counter = 0
    self.print_gap = print_gap

  def count(self, print_dict:dict=dict(), disable_print=False):
    current_time = time.time()
    if self.t0 == 0:
      self.t0 = current_time
    else:
      self.counter += 1
      if self.counter == self.print_gap:
        print_dict['FPS'] = self.counter / (current_time - self.t0)
        if not disable_print:
          for key, value in print_dict.items():
            print('{}: {}'.format(key, value), end='  ')
          print()
        self.counter = 0
        self.t0 = 0

class Glove(Thread):
  def __init__(self, data_queue:Queue):
    Thread.__init__(self)
    self.counter = Counter()
    self.data_queue = data_queue
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.sock.connect(('169.254.6.2', 11002))

  def run(self):
    while True:
      data = self.sock.recv(1024)
      if data.decode('cp437').find('VRTRIX') == 0:
        radioStrength, = struct.unpack('h', data[317:319])
        battery, = struct.unpack('f', data[319:323])
        calScore, = struct.unpack('h', data[323:325])
        raw_data = [struct.unpack('f', data[9 + 4 * i: 9 + 4 * i + 4])[0] for i in range(77)]
        self.counter.count({'RadioStrength': -radioStrength, 'Battery': battery, 'CalScore': calScore})
        data = IMUData(-raw_data[24] * 9.8, -raw_data[25] * 9.8, -raw_data[26] * 9.8, raw_data[21], raw_data[22], raw_data[23], 0)
        self.data_queue.put(data)

class GloveGestureDetector(Thread):
  def __init__(self, data_queue:Queue, handler):
    Thread.__init__(self)
    self.data_queue = data_queue
    self.counter = Counter()
    self.model = GestureNetCNN(num_classes=28).to('cpu')
    self.model.load_state_dict(torch.load('ring.pth'))
    self.model.eval()
    self.handler = handler
    self.gesture_name = {
      1: 'g_front',
      3: 'g_right',
      4: 'g_two',
      5: 'g_start',
      6: 'g_stop',
      7: 'g_prepare',
      14: 'g_one',
      15: 'g_gather',
      16: 'g_enemy',
    }
    print('Glove connected.')

  def run(self):
    imu_data = Window(200)
    gesture_window = Window(6)
    stable_window = Window(4)
    last_gesture_time = 0
    last_stable_time = 0
    while True:
      data = self.data_queue.get()
      imu_data.push(data)
      stable_window.push(data.acc_norm >= 9.4 and data.acc_norm <= 10.2)
      if stable_window.count() == stable_window.capacity():
        current_time = time.time()
        if current_time > last_stable_time + 0.5:
          last_stable_time = current_time
          print(imu_data.last())
          if imu_data.last().acc_y < -3:
            self.handler('s_front')
          if imu_data.last().acc_y > 3:
            self.handler('s_left')
      self.counter.count()
      
      if imu_data.full() and self.counter.counter % 10 == 0:
        input_data = imu_data.map(lambda x: x.to_numpy()).to_numpy().reshape(-1, 6).T
        input_tensor = torch.from_numpy(input_data).float().view(1, 6, 1, 200).to('cpu')
        outputs = F.softmax(self.model(input_tensor).cpu().detach(), dim=1)
        _, predictions = torch.max(outputs, 1)
        gesture, confidence = predictions[0].item(), outputs[0][predictions[0].item()].item()
        if gesture in self.gesture_name and confidence > 0.92:
          gesture_window.push(gesture)
          current_time = time.time()
          if current_time > last_gesture_time + 1.2 and gesture_window.full() and gesture_window.count(lambda x:x == gesture_window.first()) == gesture_window.capacity():
            self.handler(self.gesture_name[gesture])
            last_gesture_time = current_time
        else:
          gesture_window.push(-1)

class Gesture():
  def __init__(self, name, events, sound_file, handler, keep_order=False):
    self.name = name
    self.events = events
    self.sound_file = sound_file
    self.events_timestamp = [0 for _ in range(len(events))]
    self.last_trigger_timestamp = 0
    self.handler = handler
    self.keep_order = keep_order

  def playsound(self):
    print('[Gesture]', self.name)
    playsound.playsound(self.sound_file)

  def on_gesture_event(self, gesture):
    self.last_trigger_timestamp = time.time()

  def update(self, event):
    for i in range(len(self.events)):
      if self.events[i] == event:
        current_time = time.time()
        self.events_timestamp[i] = current_time
        if max(self.events_timestamp) - min(self.events_timestamp) < 1 and current_time - self.last_trigger_timestamp > 1:
          if not self.keep_order or (self.keep_order and all([self.events_timestamp[t] < self.events_timestamp[t + 1] for t in range(len(self.events) - 1)])):
            self.last_trigger_timestamp = current_time
            self.handler(self.name)
            Thread(target=self.playsound).start()



class GestureDetector():
  def __init__(self):
    glove_data_queue = Queue()
    glove = Glove(glove_data_queue)
    glove_detector = GloveGestureDetector(glove_data_queue, self.glove_gesture_handler)
    glove.daemon = True
    glove.start()
    glove_detector.daemon = True
    glove_detector.start()

    self.gesture_set = {
      Gesture('front', ['g_front', 's_front'], '/home/euxcet/gesture_train/sound/front.mp3', self.gesture_handler, keep_order=True),
      Gesture('left', ['g_front', 's_left'], '/home/euxcet/gesture_train/sound/left.mp3', self.gesture_handler, keep_order=True),
      Gesture('right', ['g_right'], '/home/euxcet/gesture_train/sound/right.mp3', self.gesture_handler),
      Gesture('two', ['g_two'], '/home/euxcet/gesture_train/sound/two.mp3', self.gesture_handler),
      Gesture('start', ['g_start'], '/home/euxcet/gesture_train/sound/start.mp3', self.gesture_handler),
      Gesture('stop', ['g_stop'], '/home/euxcet/gesture_train/sound/stop.mp3', self.gesture_handler),
      Gesture('prepare', ['g_prepare'], '/home/euxcet/gesture_train/sound/prepare.mp3', self.gesture_handler),
      Gesture('one', ['g_one'], '/home/euxcet/gesture_train/sound/one.mp3', self.gesture_handler),
      Gesture('gather', ['g_gather'], '/home/euxcet/gesture_train/sound/gather.mp3', self.gesture_handler),
      Gesture('enemy', ['g_enemy'], '/home/euxcet/gesture_train/sound/enemy.mp3', self.gesture_handler),
    }
    
    time.sleep(15)
    while True:
      time.sleep(1)

  def gesture_handler(self, gesture):
    for g in self.gesture_set:
      g.on_gesture_event(gesture)

  def glove_gesture_handler(self, glove_gesture):
    print('Glove', glove_gesture)
    for gesture in self.gesture_set:
      gesture.update(glove_gesture)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  args = parser.parse_args()
  GestureDetector()
