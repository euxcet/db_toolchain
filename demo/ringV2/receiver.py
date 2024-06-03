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
from db_zoo.node.device.ringV2 import RingV2Action

class Receiver(Node):

  INPUT_EDGE_IMU = 'imu'
  INPUT_EDGE_MIC = 'mic'
  INPUT_EDGE_TOUCH = 'touch'
  INPUT_EDGE_TOUCH_RAW = 'touch_raw'
  INPUT_EDGE_BATTERY = 'battery'
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
    self.running = False
    self.first_battery_time = 0
    self.first_battery = 0

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
      print('\nStart recording')
      self.output(self.OUTPUT_EDGE_ACTION, RingV2Action.OPEN_MIC)
    elif key == ',':
      print('\nEnd of recording')
      self.output(self.OUTPUT_EDGE_ACTION, RingV2Action.CLOSE_MIC)
      self.save_audio()
    elif key == 'v':
      self.first_battery = 0
      self.first_battery_time = 0
    elif key == 'b':
      self.output(self.OUTPUT_EDGE_ACTION, RingV2Action.GET_BATTERY_LEVEL)

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

  def handle_input_edge_imu(self, data: IMUData, timestamp: float) -> None:
    self.imu_counter.count(enable_print=True, print_dict={'Sensor': 'IMU'})

  def handle_input_edge_mic(self, data: tuple, timestamp: float) -> None:
    self.mic_counter.count(enable_print=True, print_dict={'Sensor': 'MIC'})
    length, seq, bytes = data
    self.audio += bytes

  def handle_input_edge_touch(self, data: tuple, timestamp: float) -> None:
    print('Event', data)

  def handle_input_edge_touch_raw(self, data: tuple, timestamp: float) -> None:
    length, bytes = data
    channel = ((bytes[1] & 0x02) > 0, (bytes[1] & 0x8) > 0, (bytes[1] & 0x20) > 0)

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
