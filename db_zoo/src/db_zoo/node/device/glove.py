from __future__ import annotations

import time
import socket
import struct
from enum import Enum
from typing_extensions import override
from db_graph.framework.graph import Graph
from db_graph.framework.device import Device, DeviceLifeCircleEvent
from db_graph.data.quaternion_data import QuaternionData
from db_graph.data.imu_data import IMUData
from db_graph.data.glove_data import GloveData

class GloveVersion(Enum):
  IMU_6AXIS = 0
  IMU_9AXIS = 1
  QUATERNION = 2
  IMU_6AXIS_QUATERNION = 3

  def from_str(s: str) -> GloveVersion:
    return GloveVersion.__members__[s]

  def __str__(self) -> str:
    return self.name

class Glove(Device):

  OUTPUT_EDGE_STATUS = 'status'
  OUTPUT_EDGE_LIFECYCLE = 'lifecycle'
  OUTPUT_EDGE_IMU = 'imu'
  OUTPUT_EDGE_QUATERNION = 'quaternion'

  def __init__(
      self,
      name: str,
      graph: Graph,
    input_edges: dict[str, str],
    output_edges: dict[str, str],
    ip: str,
    port: int = 11002,
    version: str = 'IMU_6AXIS',
      quiet_log: bool = False,
  ) -> None:
    super(Glove, self).__init__(
      name=name,
      graph=graph,
      input_edges=input_edges,
      output_edges=output_edges,
    )
    self.address = (ip, port)
    self.version = GloveVersion.from_str(version)
    self.quiet_log = quiet_log

  # lifecycle callbacks
  @override
  def on_pair(self) -> None:
    self.log_info('Connecting to %s:%s.' % self.address)
    self.lifecycle_status = DeviceLifeCircleEvent.on_pair
    self.output(self.OUTPUT_EDGE_LIFECYCLE, DeviceLifeCircleEvent.on_pair)

  @override
  def on_connect(self) -> None:
    self.log_info("Connected")
    self.lifecycle_status = DeviceLifeCircleEvent.on_connect
    self.output(self.OUTPUT_EDGE_LIFECYCLE, DeviceLifeCircleEvent.on_connect)

  @override
  def on_disconnect(self) -> None:
    self.log_info("Disconnected")
    self.lifecycle_status = DeviceLifeCircleEvent.on_disconnect
    self.output(self.OUTPUT_EDGE_LIFECYCLE, DeviceLifeCircleEvent.on_disconnect)

  @override
  def on_error(self) -> None:
    self.lifecycle_status = DeviceLifeCircleEvent.on_error
    self.output(self.OUTPUT_EDGE_LIFECYCLE, DeviceLifeCircleEvent.on_error)

  def scale_acc(self, data:tuple) -> tuple:
    return (data[3] * -9.8, data[4] * -9.8, data[5] * -9.8, data[0], data[1], data[2])

  def parse_data(self, data:bytearray) -> None:
    current_time = time.time()
    joint_imus, joint_quaternions = None, None
    if data.decode('cp437').find('VRTRIX') == 0:
      if self.version == GloveVersion.QUATERNION:
        radioStrength, battery, calScore = struct.unpack('<hfh', data[265:273])
        joint_quaternions = [QuaternionData(struct.unpack('<ffff', data[9 + 16 * i: 25 + 16 * i]), current_time) for i in range(16)]
        self.output(self.OUTPUT_EDGE_QUATERNION, GloveData(joint_quaternions=joint_quaternions).get_quaternion_data_numpy())
      elif self.version == GloveVersion.IMU_6AXIS:
        radioStrength, battery, calScore = struct.unpack('<hfh', data[317:325])
        joint_imus = [IMUData(self.scale_acc(struct.unpack('<fffffff', data[9 + 28 * i: 37 + 28 * i])), current_time) for i in range(11)]
        self.output(self.OUTPUT_EDGE_IMU, joint_imus)
      elif self.version == GloveVersion.IMU_6AXIS_QUATERNION:
        radioStrength, battery, calScore = struct.unpack('<hfh', data[573:581])
        joint_imus = [IMUData(self.scale_acc(struct.unpack('<fffffff', data[9 + 28 * i: 37 + 28 * i])), current_time) for i in range(11)]
        joint_quaternions = [QuaternionData(struct.unpack('<ffff', data[317 + 16 * i: 333 + 16 * i]), current_time) for i in range(16)]
        self.output(self.OUTPUT_EDGE_IMU, joint_imus)
        self.output(self.OUTPUT_EDGE_QUATERNION, GloveData(joint_quaternions=joint_quaternions).get_quaternion_data_numpy())
      self.output(self.OUTPUT_EDGE_STATUS, (radioStrength, battery, calScore))
  
  @override
  def connect(self) -> None:
    self.on_pair()
    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.socket.connect(self.address)
    self.on_connect()
    while True:
      data = self.socket.recv(581 if self.version == GloveVersion.IMU_6AXIS_QUATERNION else 1024)
      self.parse_data(data)

  @override
  def disconnect(self) -> None: ...

  @override
  def reconnect(self) -> None: ...
