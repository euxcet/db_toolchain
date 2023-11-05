from __future__ import annotations

import time
import asyncio
import struct
from bleak import BleakScanner, BleakClient
from .imu_data import IMUData
import queue
from enum import Enum
from utils.file_utils import load_json
from utils.logger import logger

class RingLifeCircleEvent(Enum):
  on_connect = 0
  on_disconnect = 1

class RingEventType(Enum):
  lifecircle = 0
  imu = 1
  touch = 2
  battery = 3

class RingEvent():
  def __init__(self, event_type:RingEventType, data, timestamp:float, address:str):
    self.event_type = event_type
    self.data = data
    self.address = address
    self.timestamp = timestamp

class RingConfig():
  def __init__(self, address:str, name:str="Ring Unnamed", adapter:str=None, imu_freq=200,
               enable_imu=True, enable_touch=True, quiet_log=False):
    self.address = address
    self.name = name
    self.adapter = adapter
    self.imu_freq = imu_freq
    self.enable_imu = enable_imu
    self.enable_touch = enable_touch
    self.quiet_log = quiet_log

  def load_from_file(file_path):
    return RingConfig(**load_json(file_path))

class RingBLE:
  EDPT_QUERY_SS			          = 0
  EDPT_OP_SYS_CONF            = 3
  EDPT_OP_ACTION_CLASS        = 4
  EDPT_OP_GSENSOR_STATE       = 10
  EDPT_OP_GSENSOR_CTL	        = 11
  EDPT_OP_HR_BO_STATE	        = 14
  EDPT_OP_HR_BO_CTL           = 15
  EDPT_OP_REPORT_HR_BO_LOG    = 0x11
  EDPT_OP_SYS_DEBUG_BIT       = 0x12
  EDPT_OP_TEMPERSTURE_QUERY   = 0x13
  EDPT_OP_GSENSOR_SWITCH_MODE = 0x20
  EDPT_OP_GSENSOR_DATA        = 0x21
  EDPT_OP_RESET_SYS_CONF      = 0x22
  EDPT_OP_LED_FLASH           = 0x23
  EDPT_OP_TOUCH_ACTION        = 0x24

  NOTIFY_CHARACTERISTIC = '0000FF11-0000-1000-8000-00805F9B34FB'
  SPP_READ_CHARACTERISTIC = 'A6ED0202-D344-460A-8075-B9E8EC90D71B'
  SPP_WRITE_CHARACTERISTIC = 'A6ED0203-D344-460A-8075-B9E8EC90D71B'

  def __init__(self, config:RingConfig, event_callback=None):
    self.config = config
    self.event_callback = event_callback

    self.client = None
    self.action_queue = queue.Queue()

    # state
    self.disconnected = False
    self.imu_mode = False

    self.raw_imu_data = bytearray()
    self.color = None

  def on_connected(self):
    if self.event_callback is not None:
      self.event_callback(RingEvent(RingEventType.lifecircle, RingLifeCircleEvent.on_connect, time.time(), self.config.address))

  def on_disconnect(self, clients):
    self.disconnected = True
    if self.event_callback is not None:
      self.event_callback(RingEvent(RingEventType.lifecircle, RingLifeCircleEvent.on_disconnect, time.time(), self.config.address))

  def log_info(self, message):
    if not self.config.quiet_log:
      logger.info(message)

  def ble_notify_callback(self, sender, data):
    crc = self.crc16(data)
    crc_l = crc & 0xFF
    crc_h = (crc >> 8) & 0xFF
    if crc_l != data[1] or crc_h != data[2]:
      logger.error('crc is wrong!')
      return

    if data[0] == self.EDPT_QUERY_SS:
      if self.event_callback is not None:
        self.event_callback(RingEvent(RingEventType.battery, data[15] & 0xFF, time.time(), self.config.address))
    elif data[0] == self.EDPT_OP_GSENSOR_STATE:
      union_val = ((data[6] & 0xFF) << 24) | ((data[5] & 0xFF) << 16) | ((data[4] & 0xFF) << 8) | (data[3] & 0xFF)
      chip_state = union_val & 0x1
      work_state = (union_val >> 1) & 0x1
      step_count = (union_val >> 8) & 0x00FFFFFF
      self.log_info('gsensor state {} {} {}'.format(chip_state, work_state, step_count))
    elif data[0] == self.EDPT_OP_TOUCH_ACTION:
      op_type = data[3] & 0x3
      report_path = (data[3] >> 2) & 0x3
      action_code = data[4]
      if op_type < 2:
        self.log_info('touch action method: ' + ['HID', 'BLE', 'HID & BLE'][report_path])
      elif op_type == 2:
        if self.event_callback is not None:
          self.event_callback(RingEvent(RingEventType.touch, action_code, time.time(), self.config.address))

  def spp_notify_callback(self, sender, data:bytearray):
    if self.imu_mode:
      self.raw_imu_data.extend(data)
      for i in range(len(self.raw_imu_data) - 1):
        if self.raw_imu_data[i] == 0xAA and self.raw_imu_data[i + 1] == 0x55:
          self.raw_imu_data = self.raw_imu_data[i:]
          break
      while len(self.raw_imu_data) > 36:
        imu_frame = self.raw_imu_data[:36]
        imu_data = IMUData(
          struct.unpack("f", imu_frame[4:8])[0],
          struct.unpack("f", imu_frame[8:12])[0],
          struct.unpack("f", imu_frame[12:16])[0],
          struct.unpack("f", imu_frame[16:20])[0],
          struct.unpack("f", imu_frame[20:24])[0],
          struct.unpack("f", imu_frame[24:28])[0],
          struct.unpack("Q", imu_frame[28:36])[0]
        )
        if self.event_callback is not None:
          self.event_callback(RingEvent(RingEventType.imu, imu_data, time.time(), self.config.address))
        crc = self.crc16(imu_frame, offset=4)
        if (crc & 0xFF) != imu_frame[2] or ((crc >> 8) & 0xFF) != imu_frame[3]:
          if not self.disconnected:
            self.send_action('disconnect')
          self.disconnected = True
        else:
          self.raw_imu_data = self.raw_imu_data[36:]
    else:
      data = data.decode()
      results = data.strip().split('\r\n')
      for result in results:
        if result.startswith('ACK'):
          if result == 'ACK:ENDB6AX':
            self.imu_mode = True
          self.log_info(result)
        elif result.startswith('ACC'):
          args = list(map(lambda x: x.split(':')[1], result.split(',')))
          acc_dict = {'0': '16g', '1': '8g', '2': '4g', '3': '2g'}
          gyro_dict = {'0': '2000dps', '1': '1000dps', '2': '500dps', '3': '250dps'}
          self.acc_fsr = args[0]
          self.gyro_fsr = args[1]
          self.imu_freq = float(args[3])
          self.log_info('IMU ACC FSR: {}  IMU GYRO FSY: {}  IMU FREQ: {}Hz'.format(acc_dict[args[0]], gyro_dict[args[1]], args[3]))
        else:
          self.log_info(result)

  def crc16(self, data, offset=3):
    genpoly = 0xA001
    result = 0xFFFF
    for i in range(offset, len(data)):
      result = (result & 0xFFFF) ^ (data[i] & 0xFF)
      for _ in range(8):
        if (result & 0x0001) == 1:
          result = (result >> 1) ^ genpoly
        else:
          result = result >> 1
    return result & 0xFFFF

  def check_data(self, data, type):
    crc = self.crc16(data)
    data[0] = type
    data[1] = crc & 0xFF
    data[2] = (crc >> 8) & 0xFF
    return data

  def query_system_conf(self):
    data = bytearray(4)
    return self.check_data(data, self.EDPT_OP_SYS_CONF)

  def query_hrbo_state(self):
    data = bytearray(11)
    data[9] = 10
    return self.check_data(data, self.EDPT_OP_HR_BO_STATE)

  def query_action_by_sel_bit(self, sel_bit):
    data = bytearray(6)
    sel_bit <<= 1
    data[3] = sel_bit & 0xFF
    data[4] = (sel_bit >> 8) & 0xFF
    data[5] = (sel_bit >> 16) & 0xFF
    return self.check_data(data, self.EDPT_OP_ACTION_CLASS)

  def set_debug_hrbo(self, enable):
    data = bytearray(4)
    data[3] = 0x3 if enable else 0x1
    return self.check_data(data, self.EDPT_OP_SYS_DEBUG_BIT)

  def query_power_sync_ts(self):
    data = bytearray(21)
    now_sec = int(time.time())
    data[17] = (now_sec >> 0) & 0xFF
    data[18] = (now_sec >> 8) & 0xFF
    data[19] = (now_sec >> 16) & 0xFF
    data[20] = (now_sec >> 24) & 0xFF
    return self.check_data(data, self.EDPT_QUERY_SS)

  def do_op_touch_action(self, get_or_set, path_code, action_code):
    data = bytearray(5)
    data[3] = ((path_code & 0x3) << 2) | (get_or_set & 0x3)
    data[4] = action_code
    return self.check_data(data, self.EDPT_OP_TOUCH_ACTION)

  def set_color(self, color):
    self.color = color
    self.send_action(f'LEDSET=[{self.color}]')

  def blink(self, blink_color, blink_time=0.5):
    origin_color = self.color
    self.set_color(blink_color)
    time.sleep(blink_time)
    self.set_color(origin_color)

  async def send(self, str):
    await self.client.write_gatt_char(self.SPP_WRITE_CHARACTERISTIC, bytearray(str + '\r\n', encoding='utf-8'))
    await asyncio.sleep(0.5)

  async def get_battery(self):
    await self.client.write_gatt_char(self.NOTIFY_CHARACTERISTIC, self.query_power_sync_ts())
    await asyncio.sleep(0.1)

  def send_action(self, action:str):
    self.action_queue.put(action)

  async def connect(self):
    if self.adapter is None:
      self.client = BleakClient(self.config.address)
    else:
      self.client = BleakClient(self.config.address, adapter=self.config.adapter)
    await self.client.connect()

    self.on_connected()
    self.client.set_disconnected_callback(self.on_disconnect)

    self.log_info("Start notify")
    await self.client.start_notify(self.SPP_READ_CHARACTERISTIC, self.spp_notify_callback)
    await self.client.start_notify(self.NOTIFY_CHARACTERISTIC, self.ble_notify_callback)

    # disable hid
    await self.client.write_gatt_char(self.NOTIFY_CHARACTERISTIC, self.do_op_touch_action(1, 1, 0))

    await self.send('ENSPP')
    await self.send('ENFAST')
    # touch
    await self.send('TPOPS=' + '1,1,1' if self.config.enable_touch else '0,0,0')
    # imu
    if self.config.enable_imu != None:
      await self.send('IMUARG=0,0,0,' + str(self.imu_freq))
      await self.send('ENDB6AX')
    self.set_color('B')

    while True:
      if not self.client.is_connected:
        break
      while not self.action_queue.empty():
        data = self.action_queue.get()
        if data == 'disconnect':
          await self.client.disconnect()
        if data == 'battery':
          await self.get_battery()
        else:
          await self.send(data)
      await asyncio.sleep(0.2)

async def scan_rings():
  ring_macs = []
  devices = await BleakScanner.discover()
  for d in devices:
    if d.name is not None and 'Ring' in d.name:
      print('Found ring:', d.name, d.address)
      ring_macs.append(d.address)
  return ring_macs

# TODO:

# async def connect():
#   ring_macs = await scan_rings()
#   coroutines = []
#   for i, ring_mac in enumerate(ring_macs):
#     print(f'Found Ring {i}: UUID[{ring_mac}]')
#     ring = BLERing(ring_mac, index=str(i), imu_callback=imu_callback)
#     coroutines.append(ring.connect())

#   await asyncio.gather(*coroutines)

# if __name__ == '__main__':
#   asyncio.run(connect())
