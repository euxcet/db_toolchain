import time
import queue
import socket
import struct
from enum import Enum
from utils.file_utils import load_json
from utils.logger import logger
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

class GloveEventType(Enum):
  basic = 0
  imu_6axis = 1
  imu_9axis = 2
  quaternion = 3

class GloveEvent():
  def __init__(self, event_type:GloveEventType, data, timestamp:float, address:tuple[str, int]):
    self.event_type = event_type
    self.data = data
    self.timestamp = timestamp
    self.address = address

class GloveVersion(Enum):
  imu_6axis = 0
  imu_9axis = 1
  quaternion = 2
  imu_6axis_quaternion = 3

  def from_str(version: str):
    return {
      'IMU_6AXIS': GloveVersion.imu_6axis,
      'IMU_9AXIS': GloveVersion.imu_9axis,
      'QUATERNION': GloveVersion.quaternion,
      'IMU_6AXIS_QUATERNION': GloveVersion.imu_6axis_quaternion
    }[version]

class GloveConfig():
  # version: IMU_6AXIS IMU_9AXIS QUATERNION IMU_6AXIS_QUATERNION
  def __init__(self, ip:str, port:int, name:str="Glove UNNAMED", version:str="IMU_6AXIS", quiet_log=False):
    self.ip = ip
    self.port = port
    self.name = name
    self.version = GloveVersion.from_str(version)
    self.quiet_log = quiet_log

  @property
  def address(self) -> tuple[str, int]:
    return (self.ip, self.port)

  def load_from_file(file_path):
    return GloveConfig(**load_json(file_path))

class Glove():
  def __init__(self, config:GloveConfig, event_queue:queue.Queue):
    self.config = config
    self.address = config.address
    self.event_queue = event_queue

  def log_info(self, message):
    if not self.config.quiet_log:
      logger.info(f'[Glove {self.config.name}] ' + message)

  def log_error(self, message):
    logger.error(f'[Glove {self.config.name}] ' + message)

  def trigger_event(self, event_type:GloveEventType, data, timestamp:float):
    self.event_queue.put_nowait(GloveEvent(event_type, data, timestamp, self.address))

  def parse_data(self, data):
    joint_imus, joint_quaternions = None, None
    if data.decode('cp437').find('VRTRIX') == 0:
      if self.config.version == GloveVersion.quaternion:
        radioStrength, battery, calScore = struct.unpack('hfh', data[265:273])
        joint_quaternions = [struct.unpack('ffff', data[9 + 16 * i, 25 + 16 * i]) for i in range(16)]
      elif self.config.version == GloveVersion.imu_6axis:
        radioStrength, battery, calScore = struct.unpack('hfh', data[317:325])
        joint_imus = [struct.unpack('fffffff', data[9 + 28 * i, 37 + 28 * i]) for i in range(11)]
      elif self.config.version == GloveVersion.imu_6axis_quaternion:
        radioStrength, battery, calScore = struct.unpack('hfh', data[573:581])
        joint_imus = [struct.unpack('fffffff', data[9 + 28 * i, 37 + 28 * i]) for i in range(11)]
        joint_quaternions = [struct.unpack('ffff', data[317 + 16 * i, 333 + 16 * i]) for i in range(16)]
      current_time = time.time()
      self.trigger_event(GloveEventType.basic, {'radioStrength': radioStrength, 'battery': battery, 'calScore': calScore}, current_time)
      if joint_imus is not None:
        self.trigger_event(GloveEventType.imu_6axis, joint_imus, current_time)
      if joint_quaternions is not None:
        self.trigger_event(GloveEventType.quaternion, joint_quaternions, current_time)
  
  def run(self):
    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.log_info('Connecting to %s:%s.' % self.config.address)
    self.socket.connect(self.config.address)
    self.log_info('Glove connected.')
    while True:
      data = self.socket.recv(581 if self.config.version == GloveVersion.imu_6axis_quaternion else 1024)
      self.parse_data(data)


class GlovePool():
  def __init__(self, keep_alive=True):
    self.keep_alive = keep_alive
    # key is the ip address of the glove
    self.gloves:dict[tuple[str, int], Glove] = {}
    self.handlers:dict[str, set] = {}
    self.executor = ThreadPoolExecutor(max_workers=20, thread_name_prefix='Glove')
    self.event_queue:queue.Queue[GloveEvent] = queue.Queue()
    event_distribution_thread = Thread(target=self.event_handler)
    event_distribution_thread.daemon = True
    event_distribution_thread.start()
    if self.keep_alive:
      keep_alive_thread = Thread(target=self.keep_alive_strategy)
      keep_alive_thread.daemon = True
      keep_alive_thread.start()

  def keep_alive_strategy(self):
    pass

  def add_glove(self, config:GloveConfig) -> Glove:
    if config.address in self.gloves:
      logger.warning(f'Glove[{config.ip}:{config.port}] is already added.')
      return
    glove = Glove(config, event_queue=self.event_queue)
    self.gloves[config.address] = glove
    self.handlers[config.address] = set()
    self.executor.submit(glove.run)
    return glove

  def get_glove(self, config:GloveConfig, connect_when_miss:bool=False) -> Glove:
    if config.address in self.gloves:
      return self.gloves[config.address]
    if connect_when_miss:
      return self.add_glove(config)
    logger.warning(f'The required glove[{config.ip}:{config.port}] has not been added yet, \
                     please consider using the connect_when_miss parameter.')
    return None

  def bind_glove(self, event_handler, glove:Glove=None, config:GloveConfig=None, address:tuple[str, int]=None):
    glove_address = None if address is None else address
    glove_address = glove_address if config is None else config.address
    glove_address = glove_address if glove is None else glove.address
    if glove_address is None:
      logger.error(f'The ip address for the glove is not provided.')
      raise Exception()
    if glove_address not in self.gloves:
      logger.error(f'Ring[{glove_address[0]}:{glove_address[1]}] is not in the glove set.')
      raise Exception()
    self.handlers[glove_address].add(event_handler)

  def event_handler(self):
    while True:
      event = self.event_queue.get()
      for handler in self.handlers[event.address]:
        handler(self.gloves[event.address], event)

glove_pool = GlovePool(keep_alive=True)