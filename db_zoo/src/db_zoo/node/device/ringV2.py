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
from typing_extensions import override
from db_graph.framework.graph import Graph
from db_graph.framework.device import Device, DeviceLifeCircleEvent
from db_graph.data.imu_data import IMUData
from db_graph.utils.logger import logger
from db_graph.utils.data_utils import index_sequence
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
    self.action_queue = queue.Queue()
    self.imu_byte_array = bytearray()
    self.imu_mode = False
    self.lifecycle_status = DeviceLifeCircleEvent.on_create

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
            acc_y, acc_z, acc_x,
            gyr_y, gyr_z, gyr_x,
            time.time(),
        ))
    elif data[2] == 0x61 and data[3] == 0x0:
      self.output(self.OUTPUT_EDGE_TOUCH, data[4])
    elif data[2] == 0x61 and data[3] == 0x1:
      self.output(self.OUTPUT_EDGE_TOUCH_RAW, data[4:])
    elif data[2] == 0x71 and data[3] == 0x0:
      length, seq = struct.unpack('<hi', data[4:10])
      self.output(self.OUTPUT_EDGE_MIC, (length, seq, data[10:]))

  def handle_input_edge_action(self, data: RingV2Action, timestamp: float) -> None:
    self.action_queue.put(data)

  async def _perform_action(self) -> None:
    while not self.action_queue.empty():
      action = self.action_queue.get()
      if action == RingV2Action.DISCONNECT:
        self.disconnect()
      elif action == RingV2Action.RECONNECT:
        self.reconnect()
      elif action == RingV2Action.GET_BATTERY_LEVEL:
        await self.write(NotifyProtocol.GET_BATTERY_LEVEL)
      elif action == RingV2Action.OPEN_MIC:
        await self.write(NotifyProtocol.OPEN_MIC)
      elif action == RingV2Action.CLOSE_MIC:
        await self.write(NotifyProtocol.CLOSE_MIC)
      elif action == RingV2Action.OPEN_IMU:
        await self.write(NotifyProtocol.OPEN_6AXIS_IMU)
      elif action == RingV2Action.CLOSE_IMU:
        await self.write(NotifyProtocol.CLOSE_6AXIS_IMU)