from __future__ import annotations

import time
import queue
import socket
import struct
from enum import Enum
from sensor.device import Device, DeviceLifeCircleEvent
from sensor.data.quaternion_data import QuaternionData
from sensor.data.imu_data import IMUData
from sensor.data.glove_data import GloveData
from utils.logger import logger
from utils.file_utils import load_json

class GloveVersion(Enum):
  IMU_6AXIS = 0
  IMU_9AXIS = 1
  QUATERNION = 2
  IMU_6AXIS_QUATERNION = 3

  def from_str(s: str) -> GloveVersion:
    return GloveVersion.__members__[s]

  def __str__(self) -> str:
    return self.name

class GloveStreamEnum(Enum):
  LIFECYCLE = 0
  STATUS = 1
  IMU = 2
  QUATERNION = 3

  def __str__(self) -> str:
    return self.name

class GloveConfig():
  def __init__(self, ip:str, port:int=11002, name:str="Glove UNNAMED",
               version:str="IMU_6AXIS", quiet_log=False, **kwargs) -> None:
    self.ip = ip
    self.port = port
    self.name = name
    self.version = GloveVersion.from_str(version)
    self.quiet_log = quiet_log

  @property
  def address(self) -> tuple[str, int]:
    return (self.ip, self.port)

  def load_from_file(file_path) -> GloveConfig:
    return GloveConfig(**load_json(file_path))

class Glove(Device):
  def __init__(self, config:GloveConfig) -> None:
    self.config = config
    self.address = config.address
    super(Glove, self).__init__()

  @property
  def name(self) -> str:
    return self.config.name

  @property
  def stream_names(self) -> str:
    return [name for name in GloveStreamEnum.__members__.values()]

  # lifecycle callbacks
  def on_pair(self) -> None:
    self.log_info('Connecting to %s:%s.' % self.config.address)
    self.lifecycle_status = DeviceLifeCircleEvent.on_pair
    self.produce_data(GloveStreamEnum.LIFECYCLE, DeviceLifeCircleEvent.on_pair)

  def on_connect(self) -> None:
    self.log_info("Connected")
    self.lifecycle_status = DeviceLifeCircleEvent.on_connect
    self.produce_data(GloveStreamEnum.LIFECYCLE, DeviceLifeCircleEvent.on_connect)

  def on_disconnect(self, *args, **kwargs) -> None:
    self.log_info("Disconnected")
    self.lifecycle_status = DeviceLifeCircleEvent.on_disconnect
    self.produce_data(GloveStreamEnum.LIFECYCLE, DeviceLifeCircleEvent.on_disconnect)

  def on_error(self) -> None:
    self.lifecycle_status = DeviceLifeCircleEvent.on_error
    self.produce_data(GloveStreamEnum.LIFECYCLE, DeviceLifeCircleEvent.on_error)

  def scale_acc(self, data:tuple) -> tuple:
    return (data[3] * -9.8, data[4] * -9.8, data[5] * -9.8, data[0], data[1], data[2])

  def parse_data(self, data:bytearray) -> None:
    current_time = time.time()
    joint_imus, joint_quaternions = None, None
    if data.decode('cp437').find('VRTRIX') == 0:
      if self.config.version == GloveVersion.QUATERNION:
        radioStrength, battery, calScore = struct.unpack('<hfh', data[265:273])
        joint_quaternions = [QuaternionData(struct.unpack('<ffff', data[9 + 16 * i: 25 + 16 * i]), current_time) for i in range(16)]
        self.produce_data(GloveStreamEnum.QUATERNION, GloveData(joint_quaternions=joint_quaternions).get_quaternion_data_numpy())
      elif self.config.version == GloveVersion.IMU_6AXIS:
        radioStrength, battery, calScore = struct.unpack('<hfh', data[317:325])
        joint_imus = [IMUData(self.scale_acc(struct.unpack('<fffffff', data[9 + 28 * i: 37 + 28 * i])), current_time) for i in range(11)]
        self.produce_data(GloveStreamEnum.IMU, joint_imus)
      elif self.config.version == GloveVersion.IMU_6AXIS_QUATERNION:
        radioStrength, battery, calScore = struct.unpack('<hfh', data[573:581])
        joint_imus = [IMUData(self.scale_acc(struct.unpack('<fffffff', data[9 + 28 * i: 37 + 28 * i])), current_time) for i in range(11)]
        joint_quaternions = [QuaternionData(struct.unpack('<ffff', data[317 + 16 * i: 333 + 16 * i]), current_time) for i in range(16)]
        self.produce_data(GloveStreamEnum.IMU, joint_imus)
        self.produce_data(GloveStreamEnum.QUATERNION, GloveData(joint_quaternions=joint_quaternions).get_quaternion_data_numpy())
      self.produce_data(GloveStreamEnum.STATUS, (radioStrength, battery, calScore))
  
  def connect(self) -> None:
    self.on_pair()
    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.socket.connect(self.config.address)
    self.on_connect()
    while True:
      data = self.socket.recv(581 if self.config.version == GloveVersion.IMU_6AXIS_QUATERNION else 1024)
      self.parse_data(data)

  def disconnect(self) -> None: ...

  def reconnect(self) -> None: ...
