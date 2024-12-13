import time
import wave
import numpy as np
from pynput import keyboard
from threading import Thread
from typing_extensions import override
from db_graph.utils.counter import Counter
from db_graph.framework.graph import Graph
from db_graph.framework.node import Node
from db_graph.data.imu_data import IMUData
from db_graph.utils.window import Window
from db_zoo.node.device.ringV2 import RingV2Action
import struct
import threading

class Receiver(Node):

  INPUT_EDGE_IMU = 'imu'
  INPUT_EDGE_MIC = 'mic'
  INPUT_EDGE_TOUCH = 'touch'
  INPUT_EDGE_TOUCH_RAW = 'touch_raw'
  INPUT_EDGE_BATTERY = 'battery'
  INPUT_EDGE_PPG_G = 'ppg_g'
  INPUT_EDGE_PPG_R = 'ppg_r'
  INPUT_EDGE_PPG_HR = 'ppg_hr'
  INPUT_EDGE_PPG_SPO2 = 'ppg_spo2'
  OUTPUT_EDGE_ACTION = 'action'

  def __init__(
      self,
      name: str,
      graph: Graph,
      input_edges: dict[str, str],
      output_edges: dict[str, str],
  ) -> None:
    super(Receiver, self).__init__(
      name=name,
      graph=graph,
      input_edges=input_edges,
      output_edges=output_edges,
    )
    self.imu_counter = Counter()
    self.imu_counter.print_interval = 400
    self.mic_counter = Counter()
    self.mic_counter.print_interval = 100
    self.first_battery_time = 0
    self.first_battery = 0
    self.focus = True
    self.mic_recording = True
    self.audio = bytearray()
    self.test_latency = 0
    self.ppg_file = None
    self.ppg_start = 0
    self.ppg_counter = 1
    self.zero_shift_window = Window(200)
    self.lock = threading.Lock()
    self.imu_file = None
    # self.imu_file = open("imu" + str(time.time()) + ".bin", 'wb')

  @override
  def start(self):
    listener = keyboard.Listener(on_press=self.on_press)
    listener.start()

  def test_consumption(self) -> None:
    while True:
      self.output(self.OUTPUT_EDGE_ACTION, RingV2Action.GET_BATTERY_LEVEL)
      time.sleep(60)

  def on_press(self, key) -> None:
    try:
      key = key.char
    except:
      key = key.name
    if key == 'f1':
      self.focus = not self.focus
    if not self.focus:
      return
    if key == 'i':
      print('\nStart IMU')
      self.imu_counter.reset()
      self.output(self.OUTPUT_EDGE_ACTION, RingV2Action.OPEN_IMU)
    elif key == 'o':
      print('\nStop IMU')
      self.output(self.OUTPUT_EDGE_ACTION, RingV2Action.CLOSE_IMU)
    elif key == 'm':
      self.audio = bytearray()
      self.mic_counter.reset()
      self.mic_recording = True
      print('\nStart recording')
      self.output(self.OUTPUT_EDGE_ACTION, RingV2Action.OPEN_MIC)
    elif key == ',':
      print('\nEnd of recording')
      self.mic_recording = False
      self.output(self.OUTPUT_EDGE_ACTION, RingV2Action.CLOSE_MIC)
      self.save_audio()
    elif key == 'l':
      self.test_latency = time.time()
    elif key == 'v':
      self.first_battery = 0
      self.first_battery_time = 0
    elif key == 'b':
      self.output(self.OUTPUT_EDGE_ACTION, RingV2Action.GET_BATTERY_LEVEL)
    elif key == 'a':
      for data in RingV2Action.set_led_nonlinear(
        wave = [100, 3000, 8000, 10000, 10000, 8000, 3000, 100],
        red = True,
        green = False,
        blue = True,
        pwd_max = 10000,
        num_repeat = 20,
        num_play = 3,
        play_flag = 1,
      ):
        self.output(self.OUTPUT_EDGE_ACTION, data)
    elif key == 's': #呼吸灯
      self.output(self.OUTPUT_EDGE_ACTION, RingV2Action.set_led_linear(
        red = False,
        green = True,
        blue = False,
        pwd_max = 10000,
        num_repeat = 1,
        num_play = 3,
        play_flag = 1,
        sequence_len = 100,
        sequence_dir = 2,
      ))
    elif key == 'p':
      self.ppg_file = open('ppg_r.txt', 'w')
      self.output(self.OUTPUT_EDGE_ACTION, RingV2Action.open_ppg_red(time=120, freq=100, ))
      # self.output(self.OUTPUT_EDGE_ACTION, RingV2Action.OPEN_PPG_R)
    elif key == '[':
      self.lock.acquire()
      self.ppg_start = 0
      if self.ppg_file is not None:
        self.ppg_file.close()
        self.ppg_file = None
      self.output(self.OUTPUT_EDGE_ACTION, RingV2Action.CLOSE_PPG_R)
      self.lock.release()
    elif key == '0':
      self.ppg_file = open('ppg_g.txt', 'w')
      self.output(self.OUTPUT_EDGE_ACTION, RingV2Action.open_ppg_green(time=120, freq=100,))
      # self.output(self.OUTPUT_EDGE_ACTION, RingV2Action.OPEN_PPG_G)
    elif key == '-':
      self.lock.acquire()
      self.ppg_start = 0
      if self.ppg_file is not None:
        self.ppg_file.close()
        self.ppg_file = None
      self.output(self.OUTPUT_EDGE_ACTION, RingV2Action.CLOSE_PPG_G)
      self.lock.release()
    elif key == '5':
      self.output(self.OUTPUT_EDGE_ACTION, RingV2Action.GET_TOUCH_SEN)
    elif key == '6':
      self.focus = False
      s = input('Sensibility:')
      s = s.strip().split('#')[-1]
      x, y, z = list(map(int, s.split(',')))
      self.output(self.OUTPUT_EDGE_ACTION, RingV2Action.set_touch_sen(x, y, z))
      self.focus = True
    
  def count_ppg(self):
    if self.ppg_start == 0:
      self.ppg_start = time.time()
      self.ppg_counter = 0
    else:
      self.ppg_counter += 1
      if self.ppg_counter % 100 == 0:
        self.log_info(f'PPG freq: {self.ppg_counter / (time.time() - self.ppg_start)}')

  def handle_input_edge_ppg_hr(self, data, timestamp: float) -> None:
    print('hr', data)

  def handle_input_edge_ppg_spo2(self, data, timestamp: float) -> None:
    print('spo2', data)

  def handle_input_edge_ppg_g(self, data, timestamp: float) -> None:
    # print('data', data)
    self.lock.acquire()
    if self.ppg_file is not None:
      self.count_ppg()
      self.ppg_file.write(f'{data}\n')
      self.ppg_file.flush()
    self.lock.release()

  def handle_input_edge_ppg_r(self, data, timestamp: float) -> None:
    # print('data', data)
    # print(data[0], data[1], data[2])
    self.lock.acquire()
    if self.ppg_file is not None:
      self.count_ppg()
      self.ppg_file.write(f'{data[0]} {data[1]}\n')
      self.ppg_file.flush()
    self.lock.release()

  def handle_input_edge_battery(self, data: float, timestamp: float) -> None:
    if data[0] == 0:
      battery = data[1]
      self.log_info(f'Battery Level = {battery}')
      if self.first_battery_time == 0:
        self.first_battery_time = time.time()
        self.first_battery = battery
      else:
        self.log_info(f'Battery Cost (%/h) = {round((battery - self.first_battery) / (time.time() - self.first_battery_time) * 3600, 2)}')
    elif data[0] == 1:
      self.log_info(f'Battery Status = {data[1]}')

  def cal_zero_shift(self, data: IMUData) -> None:
    self.zero_shift_window.push(data)
    if self.zero_shift_window.full():
      stable = True
      for index, data in enumerate(self.zero_shift_window.window):
        if data.gyr_norm > 1:
          stable = False
        if index > 0:
          dlt = data.sub(self.zero_shift_window.window[0])
          if dlt.acc_norm > 0.3 or dlt.gyr_norm > 0.3:
            stable = False
      if stable:
        self.drift = self.zero_shift_window.last()
        print(self.drift)

  def handle_input_edge_imu(self, data: IMUData, timestamp: float) -> None:
    if self.imu_file != None:
      self.imu_file.write(struct.pack('ffffff', data.acc_x, data.acc_y, data.acc_z, data.gyr_x, data.gyr_y, data.gyr_z))
      self.imu_file.flush()
    # print(data)
    self.imu_counter.count(enable_print=True, print_dict={'Sensor': 'IMU'})

  def handle_input_edge_mic(self, data, timestamp: float) -> None:
    self.mic_counter.count(enable_print=True, print_dict={'Sensor': 'MIC'})
    self.audio += data
    for i in range(0, len(data), 2):
      v = struct.unpack('<h', data[i: i + 2])[0]
      if abs(v) > 4000 and self.test_latency > 0:
        print('Latency:', str(round(time.time() - self.test_latency, 2)) + 's')
        self.test_latency = 0

  # def handle_input_edge_mic(self, data: tuple, timestamp: float) -> None:
  #   self.mic_counter.count(enable_print=True, print_dict={'Sensor': 'MIC'})
  #   length, seq, bytes = data
  #   self.audio += bytes
  #   for i in range(0, len(bytes), 2):
  #     v = struct.unpack('<h', bytes[i: i + 2])[0]
  #     if abs(v) > 4000 and self.test_latency > 0:
  #       print('Latency:', str(round(time.time() - self.test_latency, 2)) + 's')
  #       self.test_latency = 0

  def handle_input_edge_touch(self, data: tuple, timestamp: float) -> None:
    ...
    # print('Touch Event', data)

  def handle_input_edge_touch_raw(self, data: tuple, timestamp: float) -> None:
    pass
    # length, bytes = data
    # channel = ((bytes[1] & 0x02) > 0, (bytes[1] & 0x8) > 0, (bytes[1] & 0x20) > 0)

  def save_audio(self):
    try:
      wavfile = wave.open('result.wav', 'wb')
      wavfile.setnchannels(1)
      wavfile.setsampwidth(2)
      wavfile.setframerate(8000)
      wavfile.writeframes(self.audio)
      wavfile.close()
    except Exception as e:
      print(e)
