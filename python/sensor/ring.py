from __future__ import annotations
import time
import queue
import struct
import asyncio
import inspect
from enum import Enum
from typing import Any
from bleak import BleakClient
from sensor.data import IMUData
from utils.crc import crc16
from utils.logger import logger
from utils.file_utils import load_json
from utils.data_utils import index_sequence
from sensor.device import Device, DeviceLifeCircleEvent

class RingAction(Enum):
  DISCONNECT = 0
  RECONNECT = 1
  SPP_WRITE = 2
  SET_LED_COLOR = 3
  LED_BLINK = 4
  GET_BATTERY = 5

class RingConfig():
  # TODO remove kwargs?
  def __init__(self, address:str, name:str="Ring Unnamed", adapter:str=None, imu_freq:int=200,
               enable_imu:bool=True, enable_touch:bool=True, quiet_log:bool=False, led_color:str='B', **kwargs) -> None:
    self.address = address
    self.name = name
    self.adapter = adapter
    self.imu_freq = imu_freq
    self.enable_imu = enable_imu
    self.enable_touch = enable_touch
    self.quiet_log = quiet_log
    self.led_color = led_color

  def load_from_file(file_path) -> RingConfig:
    return RingConfig(**load_json(file_path))

class NotifyProtocol():
  # EDPT
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

  # SSP
  SSP_ENSPP_RESPONSE = 'ACK:ENSPP'
  SSP_DISPP_RESPONSE = 'ACK:DISPP'
  SSP_DIFAST_RESPONSE = 'ACK:DIFAST'
  SPP_QUYCPS_RESPONSE = 'TODO' # TODO
  SPP_ENDBTP_RESPONSE = 'TODO' # TODO
  SPP_DIDBTP_RESPONSE = 'ACK:DIDBTP'
  SPP_ENDB6AX_RESPONSE = 'ACK:ENDB6AX'
  SPP_DIDB6AX_RESPONSE = 'ACK:DIDB6AX'
  SPP_ENDBLOG_RESPONSE = 'ACK:ENDBLOG'
  SPP_DIDBLOG_RESPONSE = 'ACK:DIDBLOG'
  SPP_REBOOT_RESPONSE = ''
  SPP_QUERYS_RESPONSE = 'TODO' # TODO
  SPP_TPARG_RESPONSE = 'TODO' # TODO
  SPP_IMUARG_RESPONSE = 'SET IMUARG OK'
  SPP_DEVINFO_RESPONSE = 'SET DEVINFO OK'
  SPP_LEDSET_RESPONSE = 'LEDSET OK'
  SPP_TPOPS_RESPONSE = 'TPOPS SET OK'

  # CHARACTERISTIC
  NOTIFY_CHARACTERISTIC = '0000FF11-0000-1000-8000-00805F9B34FB'
  SPP_READ_CHARACTERISTIC = 'A6ED0202-D344-460A-8075-B9E8EC90D71B'
  SPP_WRITE_CHARACTERISTIC = 'A6ED0203-D344-460A-8075-B9E8EC90D71B'

  def fill_crc(data:bytearray, type:int) -> bytearray:
    crc = crc16(data)
    data[0:3] = [type, crc & 0xFF, (crc >> 8) & 0xFF]
    return data

  def check_crc(data:bytearray, offset:int, crc_offset:int) -> bool:
    crc = crc16(data, offset)
    if crc & 0xFF != data[crc_offset] or (crc >> 8) & 0xFF != data[crc_offset + 1]:
      logger.error('CRC(cyclic Redundancy Check) is wrong.')
      return False
    return True

  def query_system_conf() -> bytearray:
    data = bytearray(4)
    return NotifyProtocol.fill_crc(data, NotifyProtocol.EDPT_OP_SYS_CONF)

  def query_hrbo_state() -> bytearray:
    data = bytearray(11)
    data[9] = 10
    return NotifyProtocol.fill_crc(data, NotifyProtocol.EDPT_OP_HR_BO_STATE)

  def query_action_by_sel_bit(sel_bit) -> bytearray:
    data = bytearray(6)
    sel_bit <<= 1
    data[3] = sel_bit & 0xFF
    data[4] = (sel_bit >> 8) & 0xFF
    data[5] = (sel_bit >> 16) & 0xFF
    return NotifyProtocol.fill_crc(data, NotifyProtocol.EDPT_OP_ACTION_CLASS)

  def query_power_sync_ts() -> bytearray:
    data = bytearray(21)
    now_sec = int(time.time())
    data[17] = (now_sec >> 0) & 0xFF
    data[18] = (now_sec >> 8) & 0xFF
    data[19] = (now_sec >> 16) & 0xFF
    data[20] = (now_sec >> 24) & 0xFF
    return NotifyProtocol.fill_crc(data, NotifyProtocol.EDPT_QUERY_SS)

  def set_debug_hrbo(enable) -> bytearray:
    data = bytearray(4)
    data[3] = 0x3 if enable else 0x1
    return NotifyProtocol.fill_crc(data, NotifyProtocol.EDPT_OP_SYS_DEBUG_BIT)

  def do_op_touch_action(get_or_set, path_code, action_code) -> bytearray:
    data = bytearray(5)
    data[3] = ((path_code & 0x3) << 2) | (get_or_set & 0x3)
    data[4] = action_code
    return NotifyProtocol.fill_crc(data, NotifyProtocol.EDPT_OP_TOUCH_ACTION)

  def decode_notify_callback(data:bytearray) -> tuple:
    NotifyProtocol.check_crc(data, offset=3, crc_offset=1)
    if data[0] == NotifyProtocol.EDPT_QUERY_SS:
      return (data[0], data[15] & 0xFF)
    elif data[0] == NotifyProtocol.EDPT_OP_GSENSOR_STATE:
      union_val = ((data[6] & 0xFF) << 24) | ((data[5] & 0xFF) << 16) | ((data[4] & 0xFF) << 8) | (data[3] & 0xFF)
      return (data[0], union_val & 0x1, (union_val >> 1) & 0x1, (union_val >> 8) & 0x00FFFFFF) # chip_state, work_state, step_count
    elif data[0] == NotifyProtocol.EDPT_OP_TOUCH_ACTION:
      return (data[0], data[3] & 0x3, (data[3] >> 2) & 0x3, data[4]) # op_type, report_method, action_code

  def decode_spp_notify_callback(data:bytearray, imu_mode:bool, imu_byte_buffer:bytearray=None) -> tuple[list, bytearray]:
    if imu_mode:
      imu_byte_buffer.extend(data)
      mark_pos = index_sequence(imu_byte_buffer, [0xAA, 0x55])
      if mark_pos is not None:
        imu_byte_buffer = imu_byte_buffer[mark_pos:]
      imu_data = []
      while len(imu_byte_buffer) > 36:
        NotifyProtocol.check_crc(imu_byte_buffer[:36], offset=4, crc_offset=2)
        imu_data.append(IMUData(
          # TODO use >ffffff
          struct.unpack("f", imu_byte_buffer[4:8])[0],
          struct.unpack("f", imu_byte_buffer[8:12])[0],
          struct.unpack("f", imu_byte_buffer[12:16])[0],
          struct.unpack("f", imu_byte_buffer[16:20])[0],
          struct.unpack("f", imu_byte_buffer[20:24])[0],
          struct.unpack("f", imu_byte_buffer[24:28])[0],
          (float)(struct.unpack("Q", imu_byte_buffer[28:36])[0])
        ))
        del imu_byte_buffer[:36]
      return imu_data, imu_byte_buffer
    else:
      decoded = []
      for result in data.decode().strip().split('\r\n'):
        if result.startswith('ACC'):
          args = list(map(lambda x: x.split(':')[1], result.split(',')))
          acc_conv = {'0': '16g', '1': '8g', '2': '4g', '3': '2g'}
          gyro_conv = {'0': '2000dps', '1': '1000dps', '2': '500dps', '3': '250dps'}
          decoded.append((NotifyProtocol.SPP_IMUARG_RESPONSE, acc_conv[args[0]], gyro_conv[args[1]], args[3]))
        else:
          decoded.append((result,))
      return decoded, bytearray()

class RingStreamEnum(Enum):
  IMU = 0
  TOUCH = 1
  LIFECYCLE = 2
  BATTERY = 3

  def __str__(self) -> str:
    return {
      RingStreamEnum.IMU: 'IMU',
      RingStreamEnum.TOUCH: 'TOUCH',
      RingStreamEnum.LIFECYCLE: 'LIFECYCLE',
      RingStreamEnum.BATTERY: 'BATTERY',
    }[self]

class Ring(Device):
  def __init__(self, config:RingConfig) -> None:
    self.config = config
    super(Ring, self).__init__()
    self.action_queue = queue.Queue()
    self.imu_byte_array = bytearray()
    self.imu_mode = False
    self.led_color = config.led_color
    self.lifecycle_status = DeviceLifeCircleEvent.on_create

  @property
  def address(self) -> str:
    return self.config.address

  @property
  def name(self) -> str:
    return self.config.name

  @property
  def stream_names(self) -> str:
    return [name for name in RingStreamEnum.__members__.values()]

  # lifecycle callbacks
  def on_pair(self) -> None:
    self.log_info("Pairing")
    self.lifecycle_status = DeviceLifeCircleEvent.on_pair
    self.produce_data(RingStreamEnum.LIFECYCLE, DeviceLifeCircleEvent.on_pair)

  def on_connect(self) -> None:
    self.log_info("Connected")
    self.lifecycle_status = DeviceLifeCircleEvent.on_connect
    self.produce_data(RingStreamEnum.LIFECYCLE, DeviceLifeCircleEvent.on_connect)

  def on_disconnect(self, *args, **kwargs) -> None:
    self.log_info("Disconnected")
    self.lifecycle_status = DeviceLifeCircleEvent.on_disconnect
    self.produce_data(RingStreamEnum.LIFECYCLE, DeviceLifeCircleEvent.on_disconnect)

  def on_error(self) -> None:
    self.lifecycle_status = DeviceLifeCircleEvent.on_error
    self.produce_data(RingStreamEnum.LIFECYCLE, DeviceLifeCircleEvent.on_error)

  # active control
  async def connect(self) -> None:
    self.on_pair()
    # TODO test
    if self.config.adapter is None:
      self.client = BleakClient(self.config.address, disconnected_callback=self.on_disconnect)
    else:
      self.client = BleakClient(self.config.address, disconnected_callback=self.on_disconnect, adapter=self.config.adapter)
    await self.client.connect()

    self.log_info("Start notify")
    await self.client.start_notify(NotifyProtocol.SPP_READ_CHARACTERISTIC, self.spp_notify_callback)
    await self.client.start_notify(NotifyProtocol.NOTIFY_CHARACTERISTIC, self.notify_callback)
    await self.client.write_gatt_char(NotifyProtocol.NOTIFY_CHARACTERISTIC, NotifyProtocol.do_op_touch_action(1, 1, 0))
    await self.spp_write('ENSPP')
    await self.spp_write('ENFAST')
    await self.spp_write('TPOPS=' + '1,1,1' if self.config.enable_touch else '0,0,0')
    if self.config.enable_imu:
      await self.spp_write('IMUARG=0,0,0,' + str(self.config.imu_freq))
      await self.spp_write('ENDB6AX')
    await self.set_led_color(self.led_color)
    self.on_connect()

    while self.client.is_connected:
      await self._perform_action()
      await asyncio.sleep(0.2)
  
  def connect_sync(self) -> None:
    asyncio.run(self.connect())

  async def disconnect(self) -> None:
    await self.client.disconnect()

  async def reconnect(self) -> None:
    await self.client.disconnect()
    await self.connect()

  # notify
  def notify_callback(self, sender, data:bytearray) -> None:
    result = NotifyProtocol.decode_notify_callback(data)
    if result[0] == NotifyProtocol.EDPT_QUERY_SS:
      self.produce_data(RingStreamEnum.BATTERY, result[1])
    if result[0] == NotifyProtocol.EDPT_OP_TOUCH_ACTION:
      op_type, report_method, action_code = result[1:]
      if op_type < 2:
        self.log_info('Touch action method: ' + ['HID', 'BLE', 'HID & BLE'][report_method])
      elif op_type == 2:
        self.produce_data(RingStreamEnum.TOUCH, action_code)

  def spp_notify_callback(self, sender, data:bytearray):
    results, self.imu_byte_array = NotifyProtocol.decode_spp_notify_callback(data, self.imu_mode, self.imu_byte_array)
    if self.imu_mode:
      for data in results:
        self.produce_data(RingStreamEnum.IMU, data)
    else:
      for result in results:
        if result[0] == NotifyProtocol.SPP_ENDB6AX_RESPONSE:
          self.imu_mode = True
        self.log_info(result[0])

  async def spp_write(self, str) -> None:
    if str is not None:
      await self.client.write_gatt_char(NotifyProtocol.SPP_WRITE_CHARACTERISTIC, bytearray(str + '\r\n', encoding='utf-8'))
      await asyncio.sleep(0.5) # reduce waiting time?

  async def notify_write(self, bytes) -> None:
    await self.client.write_gatt_char(NotifyProtocol.NOTIFY_CHARACTERISTIC, bytes)
    await asyncio.sleep(0.1) # reduce waiting time?

  # action
  def send_action(self, action:RingAction, *args, **kwargs) -> None:
    self.action_queue.put((action, args, kwargs))

  async def set_led_color(self, led_color) -> None:
    self.led_color = led_color
    await self.spp_write(f'LEDSET=[{led_color}]')
    
  async def led_blink(self, blink_color, blink_time) -> None:
    origin_color = self.led_color
    await self.set_led_color(blink_color)
    await asyncio.sleep(blink_time)
    await self.set_led_color(origin_color)

  async def get_battery(self):
    await self.notify_write(NotifyProtocol.query_power_sync_ts())

  async def _perform_action(self) -> None:
    func = {
      RingAction.DISCONNECT: self.disconnect,
      RingAction.RECONNECT: self.reconnect,
      RingAction.SPP_WRITE: self.spp_write,
      RingAction.SET_LED_COLOR: self.set_led_color,
      RingAction.LED_BLINK: self.led_blink,
      RingAction.GET_BATTERY: self.get_battery,
    }
    while not self.action_queue.empty():
      action, args, kwargs  = self.action_queue.get()
      logger.info(f'Perform action {action} , args: {args} kwargs: {kwargs}')
      if inspect.iscoroutinefunction(func[action]):
        await func[action](*args, **kwargs)
      else:
        func[action](*args, **kwargs)