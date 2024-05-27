import wave
import numpy as np
from pynput import keyboard
from threading import Thread
from typing_extensions import override
from db_graph.framework.graph import Graph
from db_graph.framework.node import Node
from db_graph.data.imu_data import IMUData
from db_zoo.node.device.ringV2 import RingV2Action

class Receiver(Node):

  INPUT_EDGE_IMU = 'imu'
  INPUT_EDGE_MIC = 'mic'
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
    self.counter.print_interval = 400

  @override
  def start(self):
    listener = keyboard.Listener(on_press=self.on_press)
    listener.start()

  def on_press(self, key):
    try:
      key = key.char
    except:
      key = key.name
    if key == 'i':
      self.output(self.OUTPUT_EDGE_ACTION, RingV2Action.OPEN_IMU)
    elif key == 'o':
      self.output(self.OUTPUT_EDGE_ACTION, RingV2Action.CLOSE_IMU)
    elif key == 'm':
      self.audio = bytearray()
      print('\nStart recording')
      self.output(self.OUTPUT_EDGE_ACTION, RingV2Action.OPEN_MIC)
    elif key == ',':
      print('\nEnd of recording')
      self.output(self.OUTPUT_EDGE_ACTION, RingV2Action.CLOSE_MIC)
      self.save_audio()
    elif key == 'b':
      self.output(self.OUTPUT_EDGE_ACTION, RingV2Action.GET_BATTERY_LEVEL)

  def handle_input_edge_battery(self, data: float, timestamp: float) -> None:
    if data[0] == 0:
      print('Battery Level:', data[1])
    elif data[0] == 1:
      print('Battery Status:', data[1])

  def handle_input_edge_imu(self, data: IMUData, timestamp: float) -> None:
    self.counter.count(enable_print=True)

  def handle_input_edge_mic(self, data: tuple, timestamp: float) -> None:
    length, seq, bytes = data
    self.audio += bytes

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
