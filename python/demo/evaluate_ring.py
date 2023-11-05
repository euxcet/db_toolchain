from core.imu_data import IMUData, IMUDataGroup
from matplotlib import pyplot as plt
import os
import socket
import struct
import time
from python.model.gesture_cnn import GestureNetCNN
import torch
import scipy
from queue import Queue
from python.core.ring_ble import RingBLE, scan_rings
import torch.nn.functional as F
from python.utils.window import Window
from threading import Thread
import asyncio

class Ring(Thread):
  def __init__(self, macs:list[str], data_queue:Queue):
    Thread.__init__(self)
    self.macs = macs
    self.data_queue = data_queue
    self.rings:list[RingBLE] = []
    self.initialize_rings()
    self.connected = False

  def initialize_rings(self):
    for index, mac in enumerate(self.macs):
      self.rings.append(RingBLE(mac, index=index, imu_callback=self.imu_callback))

  async def connect_rings(self):
    coroutines = [ring.connect() for ring in self.rings]
    await asyncio.gather(*coroutines)

  def imu_callback(self, index:int, data:IMUData):
    self.connected = True
    self.data_queue.put(data)

  def event_callback(self, event:str):
    if event == 'Disconnected':
      self.connected = False

  def run(self):
    asyncio.run(self.connect_rings())

class Counter():
  def __init__(self, print_gap:int=1000):
    self.t0 = 0
    self.counter = 0
    self.print_gap = print_gap

  def count(self, print_dict:dict=dict()):
    current_time = time.time()
    if self.t0 == 0:
      self.t0 = current_time
    else:
      self.counter += 1
      if self.counter == self.print_gap:
        print_dict['FPS'] = self.counter / (current_time - self.t0)
        for key, value in print_dict.items():
          print('{}: {}'.format(key, value), end='  ')
        print()
        self.counter = 0
        self.t0 = 0


class RingGestureDetector(Thread):
  def __init__(self, handler, data_queue:Queue):
    Thread.__init__(self)
    self.data_queue = data_queue
    self.model = GestureNetCNN(num_classes=28).to('cpu')
    self.model.load_state_dict(torch.load('ring.pth'))
    self.model.eval()
    self.handler = handler
    self.counter = Counter()
    self.move_name = {}
    self.gesture_name = {}
    # self.move_name = {
    #   0: 'left',
    #   1: 'forward',
    #   2: 'up',
    # }
    self.gesture_name = {
      1: 'front',
      3: 'right',
      4: 'two',
      5: 'start',
      6: 'stop',
      7: 'prepare',
      14: 'one',
      15: 'gather',
      16: 'enemy',
    }

  def run(self):
    imu_data = Window(200)
    gesture_window = Window(6)
    move_window = Window(6)
    stable_window = Window(4)
    last_gesture_time = 0
    last_move_time = 0
    last_stable_time = 0
    while True:
      # 1: y -   2: z -
      data = self.data_queue.get()
      imu_data.push(data)
      self.counter.count()
      stable_window.push(data.acc_norm >= 9.4 and data.acc_norm <= 10.2)
      if stable_window.count() == stable_window.capacity():
        current_time = time.time()
        if current_time > last_stable_time + 0.5:
          last_stable_time = current_time
          if imu_data.last().acc_y < -8:
            self.handler('s_front')
          if imu_data.last().acc_z < -8:
            self.handler('s_left')

      if imu_data.full() and self.counter.counter % 10 == 0:
        data = imu_data.map(lambda x: x.to_numpy()).to_numpy().reshape(-1, 6).T
        input_tensor = torch.from_numpy(data).float().view(1, 6, 1, 200).to('cpu')
        # outputs = self.model(input_tensor)
        outputs = F.softmax(self.model(input_tensor).cpu().detach(), dim=1)
        _, predictions = torch.max(outputs, 1)
        gesture, confidence = predictions[0].item(), outputs[0][predictions[0].item()].item()
        if gesture in self.gesture_name and confidence > 0.95:
          gesture_window.push(gesture)
          current_time = time.time()
          if current_time > last_gesture_time + 1.2 and gesture_window.full() and gesture_window.count(lambda x:x == gesture_window.first()) == gesture_window.capacity():
            self.handler(self.gesture_name[gesture])
            last_gesture_time = current_time
        else:
          gesture_window.push(-1)

        if gesture in self.move_name and confidence > 0.95:
          move_window.push(gesture)
          current_time = time.time()
          if current_time > last_move_time + 1.2 and move_window.full() and move_window.count(lambda x:x == move_window.first()) == move_window.capacity():
            self.handler("move")
            last_move_time = current_time
        else:
          move_window.push(-1)

class GestureDetector():
  def __init__(self):
    data_queue = Queue()
    Ring(['0D85597D-C82C-E839-E0E0-4776246A6398'], data_queue).start()
    RingGestureDetector(self.ring_gesture_handler, data_queue).start()

  def ring_gesture_handler(self, ring_gesture):
    print(ring_gesture)

if __name__ == '__main__':
  GestureDetector()
