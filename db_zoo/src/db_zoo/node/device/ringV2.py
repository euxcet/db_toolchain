from __future__ import annotations
from ...utils.version import check_library_version
check_library_version('bleak', '0.20.2')
import math
import time
import queue
import struct
import asyncio
import inspect
from enum import Enum
from bleak import BleakClient
from collections import deque
from typing_extensions import override
from db_graph.framework.graph import Graph
from db_graph.framework.device import Device, DeviceLifeCircleEvent
from db_graph.data.imu_data import IMUData
from db_graph.utils.logger import logger
from db_graph.utils.data_utils import index_sequence
from threading import Thread
import wave

class NotifyProtocol():
  READ_CHARACTERISTIC  = 'BAE80011-4F05-4503-8E65-3AF1F7329D1F'
  WRITE_CHARACTERISTIC = 'BAE80010-4F05-4503-8E65-3AF1F7329D1F'

  GET_TIME             = bytearray([0x00, 0x00, 0x10, 0x01])
  GET_SOFTWARE_VERSION = bytearray([0x00, 0x00, 0x11, 0x00])
  GET_HARDWARE_VERSION = bytearray([0x00, 0x00, 0x11, 0x01])
  GET_BATTERY_LEVEL    = bytearray([0x00, 0x00, 0x12, 0x00])
  GET_BATTERY_STATUS   = bytearray([0x00, 0x00, 0x12, 0x01])
  OPEN_6AXIS_IMU       = bytearray([0x00, 0x00, 0x40, 0x06])
  CLOSE_6AXIS_IMU      = bytearray([0x00, 0x00, 0x40, 0x00])
  GET_TOUCH            = bytearray([0x00, 0x00, 0x61, 0x00])
  OPEN_MIC             = bytearray([0x00, 0x00, 0x71, 0x00, 0x01])
  CLOSE_MIC            = bytearray([0x00, 0x00, 0x71, 0x00, 0x00])
  GET_NFC              = bytearray([0x00, 0x00, 0x82, 0x00])

class RingV2Action(Enum):
  DISCONNECT = 0
  RECONNECT = 1
  GET_BATTERY_LEVEL = 2
  OPEN_MIC = 3
  CLOSE_MIC = 4
  OPEN_IMU = 5
  CLOSE_IMU = 6

  def __str__(self):
    return self.name
  
  def from_str(s: str) -> RingV2Action:
    return RingV2Action.__members__[s]

  def set_led_linear(
    red: bool,
    green: bool,
    blue: bool,
    pwd_max: int,
    num_repeat: int,
    num_play: int,
    play_flag: int,
    sequence_len: int,
    sequence_dir: int,
  ) -> bytearray:
    data = bytearray([0x0, 0x0, 0x62, 0x1])
    data += bytearray([blue, red, green]) # 蓝红绿 三个颜色
    data += struct.pack('<H', pwd_max) # pwd计数最大值
    data += struct.pack('<I', num_repeat) # 周期重复次数
    data += struct.pack('<H', num_play) # 播放次数
    data += struct.pack('<B', play_flag) # 1单次 2循环
    data += struct.pack('<B', sequence_len) # 序列长度
    data += struct.pack('<B', sequence_dir) # 1单向 2双向
    return data

  def set_led_nonlinear(
    wave: list[int],
    red: bool,
    green: bool,
    blue: bool,
    pwd_max: int,
    num_repeat: int,
    num_play: int,
    play_flag: int,
    package_size: int = 200,
  ) -> list[bytearray]:
    packages = []
    num_packages = (len(wave) + package_size - 1) // package_size
    for i in range(0, len(wave), package_size):
      package_data = wave[i:i + package_size]
      package_no = i // package_size
      package_length = len(package_data)
      package = bytearray([0x0, 0x0, 0x62, 0x2])
      package += bytearray([blue, red, green]) # 蓝红绿 三个颜色
      package += struct.pack('<H', pwd_max) # pwd计数最大值
      package += struct.pack('<I', num_repeat) # 周期重复次数
      package += struct.pack('<H', num_play) # 播放次数
      package += struct.pack('<B', play_flag) # 1单次 2循环
      package += struct.pack('<BBB', num_packages, package_no, package_length)
      package += struct.pack(f'<{package_length}H', *package_data)
      packages.append(package)
    return packages

class RingV2(Device):

  INPUT_EDGE_ACTION     = 'action'
  OUTPUT_EDGE_LIFECYCLE = 'lifecycle'
  OUTPUT_EDGE_IMU       = 'imu'
  OUTPUT_EDGE_MIC       = 'mic'
  OUTPUT_EDGE_TOUCH     = 'touch'
  OUTPUT_EDGE_TOUCH_RAW = 'touch_raw'
  OUTPUT_EDGE_BATTERY   = 'battery'

  def __init__(
      self,
      name: str,
      graph: Graph,
      address: str,
      input_edges: dict[str, str] = {},
      output_edges: dict[str, str] = {},
      drift: list[float] = [0, 0, 0],
      adapter: str = None,
      imu_freq: int = 200,
      enable_imu: bool = True,
      enable_touch: bool = True,
      quiet_log: bool = False, 
      led_color: str = 'B',
  ) -> None:
    super(RingV2, self).__init__(
      name=name,
      graph=graph,
      input_edges=input_edges,
      output_edges=output_edges,
    )
    self.address = address
    self.adapter = adapter
    self.imu_freq = imu_freq
    self.enable_imu = enable_imu
    self.enable_touch = enable_touch
    self.quiet_log = quiet_log
    self.led_color = led_color
    self.taped = False
    self.drift = drift
    self.touch_type = -1
    self.tap_thread = Thread(target=self.tap_func)
    self.tap_thread.daemon = True
    self.tap_thread.start()
    self.action_queue = queue.Queue()
    self.imu_byte_array = bytearray()
    self.imu_mode = False
    self.lifecycle_status = DeviceLifeCircleEvent.on_create
    self.touch_history = []
    self.last_touch_timestmap = time.time()
    self.is_holding = False

  def tap_func(self):
    counter = -1
    while True:
      if self.taped:
        if counter != -1:
          self.output(self.OUTPUT_EDGE_TOUCH, 'double_tap')
          counter = -1
        else:
          counter = 0
        self.taped = False
      elif counter >= 0:
        counter += 1
        if counter == 25: # tap interval
          if self.touch_type == 0:
            self.output(self.OUTPUT_EDGE_TOUCH, 'tap')
          elif self.touch_type == 1:
            self.output(self.OUTPUT_EDGE_TOUCH, 'down')
          elif self.touch_type == 2:
            self.output(self.OUTPUT_EDGE_TOUCH, 'up')
          counter = -1
      time.sleep(0.01)

  # lifecycle callbacks
  @override
  def on_pair(self) -> None:
    self.log_info(f"Pairing <{self.address}>")
    self.lifecycle_status = DeviceLifeCircleEvent.on_pair
    self.output(self.OUTPUT_EDGE_LIFECYCLE, DeviceLifeCircleEvent.on_pair)

  @override
  def on_connect(self) -> None:
    self.log_info("Connected")
    self.lifecycle_status = DeviceLifeCircleEvent.on_connect
    self.output(self.OUTPUT_EDGE_LIFECYCLE, DeviceLifeCircleEvent.on_connect)

  @override
  def on_disconnect(self, *args, **kwargs) -> None:
    self.log_info("Disconnected")
    self.lifecycle_status = DeviceLifeCircleEvent.on_disconnect
    self.output(self.OUTPUT_EDGE_LIFECYCLE, DeviceLifeCircleEvent.on_disconnect)

  @override
  def on_error(self) -> None:
    self.lifecycle_status = DeviceLifeCircleEvent.on_error
    self.output(self.OUTPUT_EDGE_LIFECYCLE, DeviceLifeCircleEvent.on_error)

  async def write(self, data: bytearray) -> None:
    await self.client.write_gatt_char(NotifyProtocol.WRITE_CHARACTERISTIC, data)

  async def connect_async(self) -> None:
    self.on_pair()
    async with BleakClient(self.address, disconnected_callback=self.on_disconnect, adapter=self.adapter) as self.client:
      await self.client.connect()
      self.on_connect()
      self.log_info("Start notify")
      await self.client.start_notify(NotifyProtocol.READ_CHARACTERISTIC, self.notify_callback)
      await asyncio.sleep(1)
      await self.write(NotifyProtocol.GET_SOFTWARE_VERSION)
      await self.write(NotifyProtocol.GET_HARDWARE_VERSION)
      await self.write(NotifyProtocol.OPEN_6AXIS_IMU)

      while True:
        await self._perform_action()
        await asyncio.sleep(0.2)

  async def disconnect_async(self) -> None:
    await self.client.disconnect()

  async def reconnect_async(self) -> None:
    await self.client.disconnect()
    await self.connect()

  @override
  def connect(self) -> None:
    asyncio.run(self.connect_async())

  @override
  def disconnect(self) -> None:
    asyncio.run(self.disconnect_async())

  @override
  def reconnect(self) -> None:
    asyncio.run(self.reconnect_async())

  def notify_callback(self, sender, data:bytearray) -> None:
    if data[2] == 0x62 and data[3] == 0x1:
      print('呼吸灯结果')
    if data[2] == 0x62 and data[3] == 0x2:
      print('自定义灯结果')
    if data[2] == 0x62 and data[3] == 0x3:
      print('自定义灯pwm空闲结果')

    if data[2] == 0x10 and data[3] == 0x0:
      print(data[4])
    if data[2] == 0x11 and data[3] == 0x0:
      print('Software version:', data[4:])
    if data[2] == 0x11 and data[3] == 0x1:
      print('Hardware version:', data[4:])
    elif data[2] == 0x12 and data[3] == 0x0:
      self.output(self.OUTPUT_EDGE_BATTERY, (0, data[4]))
    elif data[2] == 0x12 and data[3] == 0x1:
      self.output(self.OUTPUT_EDGE_BATTERY, (1, data[4]))
    elif data[2] == 0x40 and data[3] == 0x06:
      for index in range(5, len(data), 12):
        acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z = struct.unpack('<hhhhhh', data[index:index+12])
        acc_x, acc_y, acc_z = acc_x / 1000 * 9.8, acc_y / 1000 * 9.8, acc_z / 1000 * 9.8
        gyr_x, gyr_y, gyr_z = gyr_x / 180 * math.pi,  gyr_y / 180 * math.pi, gyr_z / 180 * math.pi
        self.output(self.OUTPUT_EDGE_IMU, IMUData(
            -acc_y, acc_z, -acc_x,
            -gyr_y - self.drift[0], gyr_z - self.drift[1], -gyr_x - self.drift[2],
            time.time(),
        ))
    elif data[2] == 0x61 and data[3] == 0x0:
      ...
      # self.output(self.OUTPUT_EDGE_TOUCH, data[4])
    elif data[2] == 0x61 and data[3] == 0x1:
      self._detect_touch_events(data[5:])
      self.output(self.OUTPUT_EDGE_TOUCH_RAW, (data[4], data[5:]))
    elif data[2] == 0x71 and data[3] == 0x0:
      length, seq = struct.unpack('<hi', data[4:10])
      self.output(self.OUTPUT_EDGE_MIC, (length, seq, data[10:]))

  def handle_input_edge_action(self, data: RingV2Action|bytearray, timestamp: float) -> None:
    self.action_queue.put(data)

  def get_touch_state(self, x, y, z):
    # 0: 000 -> 0
    # 1: 001 -> 1
    # 2: 010 -> 3
    # 3: 011 -> 2
    # 4: 100 -> 5
    # 5: 101 -> -1
    # 6: 110 -> 4
    # 7: 111 -> -2
    return [0, 1, 3, 2, 5, -1, 4, -2][x * 4 + y * 2 + z]

  def _detect_touch_events(self, data: bytearray) -> None:
    new_touch = [
      self.get_touch_state(
        1 if data[1] & 0x02 else 0,
        1 if data[1] & 0x08 else 0,
        1 if data[1] & 0x20 else 0),
      time.time(), ]
    print(new_touch)
    self.touch_history.append(new_touch)
    if new_touch[0] == 0:
      if not self.is_holding and len(self.touch_history) > 1:
        self.taped = True
        if self.touch_history[-2][-1] - self.touch_history[0][-1] < 0.15:
          self.touch_type = 0
        elif self.touch_history[-2][0] > self.touch_history[0][0]:
          self.touch_type = 1
        elif self.touch_history[-2][0] < self.touch_history[0][0]:
          self.touch_type = 2
        else:
          self.touch_type = 0
      self.is_holding = False
      self.touch_history.clear()
    else:
      if self.touch_history[-1][-1] - self.touch_history[0][-1] > 1.5 and not self.is_holding:
        self.output(self.OUTPUT_EDGE_TOUCH, 'hold')
        self.is_holding = True

    # timestamp = time.time()
    # if timestamp - self.last_touch_timestmap > 1:
    #   self.touch_history.clear()
    # self.touch_history.append(new_touch)

    # status = " touch status:"
    # status += f"  {1 if data[1] & 0x02 else 0}   "
    # status += f"  {1 if data[1] & 0x08 else 0}   "
    # status += f"  {1 if data[1] & 0x20 else 0}   "
    # self.output(self.OUTPUT_EDGE_TOUCH, status)
    # if data[2] & 0x01:
    #   self.output(self.OUTPUT_EDGE_TOUCH, "tap")
    # elif data[2] & 0x02:
    #   self.output(self.OUTPUT_EDGE_TOUCH, "swipe_positive")
    # elif data[2] & 0x04:
    #   self.output(self.OUTPUT_EDGE_TOUCH, "swipe_negative")
    # elif data[2] & 0x08:
    #   self.output(self.OUTPUT_EDGE_TOUCH, "flick_positive")
    # elif data[2] & 0x10:
    #   self.output(self.OUTPUT_EDGE_TOUCH, "flick_negative")
    # elif data[2] & 0x20:
    #   self.output(self.OUTPUT_EDGE_TOUCH, "hold")

  async def _perform_action(self) -> None:
    while not self.action_queue.empty():
      action = self.action_queue.get()
      if type(action) is bytearray:
        await self.write(action)
      elif type(action) is RingV2Action:
        if action == RingV2Action.DISCONNECT:
          self.disconnect()
        elif action == RingV2Action.RECONNECT:
          self.reconnect()
        elif action == RingV2Action.GET_BATTERY_LEVEL:
          await self.write(NotifyProtocol.GET_BATTERY_LEVEL)
        # elif action == RingV2Action.OPEN_MIC:
        #   await self.write(NotifyProtocol.OPEN_MIC)
        elif action == RingV2Action.CLOSE_MIC:
          await self.write(NotifyProtocol.CLOSE_MIC)
        elif action == RingV2Action.OPEN_IMU:
          await self.write(NotifyProtocol.OPEN_6AXIS_IMU)
        elif action == RingV2Action.CLOSE_IMU:
          await self.write(NotifyProtocol.CLOSE_6AXIS_IMU)

# up 9.8 0 0
# down -9.8 0 0
# forward 0 0 -9.8
# to_left 0 -9.8 0
