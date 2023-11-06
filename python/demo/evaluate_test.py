import os
import time
import torch
import scipy
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
import playsound
import asyncio
from utils.counter import Counter


class Ring(Thread):
  def __init__(self, macs:list[str], data_queue:Queue, adapter:str):
    Thread.__init__(self)
    self.macs = macs
    self.data_queue = data_queue
    self.rings:list[RingBLE] = []
    self.adapter = adapter
    self.initialize_rings()
    self.connected = False
    self.last_data_time = 0
    self.known_rings = {
      'AA:55:AA:55:CD:DC': '0',
    }

  def initialize_rings(self):
    for index, mac in enumerate(self.macs):
      self.rings.append(RingBLE(mac, index=index, imu_callback=self.imu_callback, adapter=self.adapter, event_callback=self.event_callback))

  async def connect_rings(self):
    coroutines = [ring.connect() for ring in self.rings]
    await asyncio.gather(*coroutines)

  def imu_callback(self, index:int, data:IMUData):
    if not self.connected and self.macs[index] in self.known_rings:
      for ring in self.rings:
        ring.send_action('LEDSET=[R]')
      Thread(target=playsound.playsound('/home/euxcet/gesture_train/sound/ring' + self.known_rings[self.macs[index]]  + '.mp3')).start()
      
    self.connected = True
    self.last_data_time = time.time()
    self.data_queue.put(data)

  def is_data_flowing(self):
    current_time = time.time()
    return self.connected and current_time - self.last_data_time < 1.0

  def event_callback(self, event:str):
    if event == 'Disconnected':
      self.connected = False

  def run(self):
    asyncio.run(self.connect_rings())

  def blink_(self):
    for ring in self.rings:
      ring.send_action('LEDSET=[RGB]')
      time.sleep(0.5)
      ring.send_action('LEDSET=[R]')

  def blink(self):
    Thread(target=self.blink_).start()


class RingGestureDetector(Thread):
  def __init__(self, handler, data_queue:Queue):
    Thread.__init__(self)
    self.data_queue = data_queue
    self.model = GestureNetCNN(num_classes=28).to('cpu')
    self.model.load_state_dict(torch.load('/home/euxcet/gesture_train/ring.pth'))
    self.model.eval()
    self.handler = handler
    self.counter = Counter()
    self.move_name = {
      0: 'r_left',
      1: 'r_forward',
      2: 'r_up',
    }
    self.gesture_name = {
      3: 'r_bomb',
      4: 'r_close',
      5: 'r_forward',
      6: 'r_gather',
      7: 'r_quick',
      8: 'r_enemy',
      9: 'r_cover',
      10: 'r_repeat',
    }

  def run(self):
    imu_data = Window(200)
    gesture_window = Window(6)
    move_window = Window(6)
    last_gesture_time = 0
    last_move_time = 0
    while True:
      data = self.data_queue.get()
      imu_data.push(data)
      self.counter.count()
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
            self.handler("r_move")
            last_move_time = current_time
        else:
          move_window.push(-1)


class GloveGestureDetector(Thread):
  def __init__(self, handler, out_ip:str=None):
    Thread.__init__(self)
    self.counter = Counter()
    self.model = FullyConnectedModel(11)
    self.model.load_state_dict(torch.load('/home/euxcet/gesture_train/glove.pth'))
    self.model.eval()
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.sock.connect(('127.0.0.1', 11002))
    self.handler = handler
    self.out_sock = None

    self.gesture_name = {
      0: 'g_sniper',
      1: 'g_hostage',
      2: 'g_gather',
      3: 'g_quick',
      4: 'g_stop',
      5: 'g_search',
      6: 'g_ok',
      7: 'g_safe',
      8: 'g_danger',
      9: 'g_prepare',
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

class Gesture():
  def __init__(self, name, events, sound_file, handler):
    self.name = name
    self.events = events
    self.sound_file = sound_file
    self.events_timestamp = [0 for _ in range(len(events))]
    self.last_trigger_timestamp = 0
    self.handler = handler

  def playsound(self):
    print('[Gesture]', self.name)
    self.handler(self.name)
    playsound.playsound(self.sound_file)

  def update(self, event):
    for i in range(len(self.events)):
      if self.events[i] == event:
        current_time = time.time()
        self.events_timestamp[i] = current_time
        if max(self.events_timestamp) - min(self.events_timestamp) < 1 and current_time - self.last_trigger_timestamp > 1:
          self.last_trigger_timestamp = current_time
          Thread(target=self.playsound).start()

class GestureDetector():
  def __init__(self, adapter, ring):
    print('Ring:', ring)
    ring_data_queue = Queue()
    ring = Ring([ring], ring_data_queue, adapter)
    ring_detector = RingGestureDetector(self.ring_gesture_handler, ring_data_queue)
    glove_detector = GloveGestureDetector(self.glove_gesture_handler)

    ring.setDaemon(True)
    ring_detector.setDaemon(True)
    glove_detector.setDaemon(True)

    ring.start()
    ring_detector.start()
    glove_detector.start()

    self.ring = ring

    self.gesture_set = {
      Gesture('bomb', ['r_bomb'], '/home/euxcet/gesture_train/sound/bomb.mp3', self.gesture_handler),
      Gesture('close', ['r_close'], '/home/euxcet/gesture_train/sound/close.mp3', self.gesture_handler),
      Gesture('cover', ['r_cover'], '/home/euxcet/gesture_train/sound/cover.mp3', self.gesture_handler),
      Gesture('enemy', ['r_enemy'], '/home/euxcet/gesture_train/sound/enemy.mp3', self.gesture_handler),
      Gesture('gather', ['r_gather'], '/home/euxcet/gesture_train/sound/gather.mp3', self.gesture_handler),
      Gesture('quick', ['r_quick', 'g_quick'], '/home/euxcet/gesture_train/sound/quick.mp3', self.gesture_handler),
      Gesture('forward', ['r_forward'], '/home/euxcet/gesture_train/sound/forward.mp3', self.gesture_handler),
      Gesture('repeat', ['r_repeat'], '/home/euxcet/gesture_train/sound/repeat.mp3', self.gesture_handler),
      Gesture('hostage', ['r_move', 'g_hostage'], '/home/euxcet/gesture_train/sound/hostage.mp3', self.gesture_handler),
      Gesture('prepare', ['r_move', 'g_prepare'], '/home/euxcet/gesture_train/sound/prepare.mp3', self.gesture_handler),
      Gesture('ok', ['r_move', 'g_ok'], '/home/euxcet/gesture_train/sound/ok.mp3', self.gesture_handler),
      Gesture('safe', ['r_move', 'g_safe'], '/home/euxcet/gesture_train/sound/safe.mp3', self.gesture_handler),
      Gesture('danger', ['r_move', 'g_danger'], '/home/euxcet/gesture_train/sound/danger.mp3', self.gesture_handler),
      Gesture('search', ['r_move', 'g_search'], '/home/euxcet/gesture_train/sound/search.mp3', self.gesture_handler),
      Gesture('sniper', ['r_move', 'g_sniper'], '/home/euxcet/gesture_train/sound/sniper.mp3', self.gesture_handler),
      Gesture('stop', ['r_move', 'g_stop'], '/home/euxcet/gesture_train/sound/stop.mp3', self.gesture_handler),
    }
    
    time.sleep(15)
    while True:
      if not ring.is_data_flowing():
        break
      time.sleep(1)

  def gesture_handler(self, gesture):
    self.ring.blink()

  def glove_gesture_handler(self, glove_gesture):
    print('Glove', glove_gesture)
    '''
    current_time = time.time()
    if glove_gesture in ['hostage', 'prepare', 'ok', 'safe', 'danger', 'search', 'sniper', 'stop']:
      if current_time < self.last_move_t + 1:
        Thread(target=self.playsound, args=(glove_gesture,)).start()
      else:
        self.last_glove_t = current_time
        self.last_glove = glove_gesture
    '''
    for gesture in self.gesture_set:
        gesture.update(glove_gesture)

  def ring_gesture_handler(self, ring_gesture):
    print('Ring', ring_gesture)
    '''
    current_time = time.time()
    if ring_gesture in ['bomb', 'close', 'cover', 'enemy', 'gather', 'quick', 'forward', 'repeat']:
      Thread(target=self.playsound, args=(ring_gesture,)).start()
    if ring_gesture == 'move':
      if current_time < self.last_glove_t + 1:
        Thread(target=self.playsound, args=(self.last_glove,)).start()
      else:
        self.last_move_t = current_time
    '''
    for gesture in self.gesture_set:
        gesture.update(ring_gesture)

def disconnect_rings():
  devices = os.popen('bluetoothctl paired-devices').readlines()
  for device in devices:
    mac = device.split(' ')[1]
    os.system('bluetoothctl remove ' + mac)
    os.system('bluetoothctl remove ' + mac)

def get_ring():
  rings = asyncio.run(scan_rings())
  max_rssi = -1000
  ring_address = None
  for ring in rings:
    if ring.rssi > max_rssi:
      max_rssi = ring.rssi
      ring_address = ring.address
  return ring_address

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--adapter', type=str, default='hci1')
  parser.add_argument('--ring', type=str, default='')
  args = parser.parse_args()
  disconnect_rings()
  ring = get_ring() if args.ring == '' else args.ring
  if ring is not None:
    GestureDetector(args.adapter, ring)
