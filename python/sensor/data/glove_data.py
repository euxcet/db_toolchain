import numpy as np
from enum import Enum
from .basic_data import IMUData, QuaternionData

class GloveQuaternionJointName(Enum):
  WRIST_JOINT = 0
  THUMB_PROXIMAL = 1
  THUMB_INTERMEDIATE = 2
  THUMB_DISTAL = 3
  INDEX_PROXIMAL = 4
  INDEX_INTERMEDIATE = 5
  INDEX_DISTAL = 6
  MIDDLE_PROXIMAL = 7
  MIDDLE_INTERMEDIATE = 8
  MIDDLE_DISTAL = 9
  RING_PROXIMAL = 10
  RING_INTERMEDIATE = 11
  RING_DISTAL = 12
  PINKY_PROXIMAL = 13
  PINKY_INTERMEDIATE = 14
  PINKY_DISTAL = 15

class GloveIMUJointName(Enum):
  WRIST_JOINT = 0
  THUMB_INTERMEDIATE = 1
  THUMB_DISTAL = 2
  INDEX_PROXIMAL = 3
  INDEX_INTERMEDIATE = 4
  MIDDLE_PROXIMAL = 5
  MIDDLE_INTERMEDIATE = 6
  RING_PROXIMAL = 7
  RING_INTERMEDIATE = 8
  PINKY_PROXIMAL = 9
  PINKY_INTERMEDIATE = 10

class GloveData():
  # TODO: use a class to store basic information.
  def __init__(self, basic:dict, imu_data:list[IMUData]=None, quaternion_data:list[QuaternionData]=None, timestamp:float=0):
    self.basic = basic
    self.imu_data:list[IMUData] = imu_data
    self.quaternion_data:list[QuaternionData] = quaternion_data
    self.timestamp = timestamp

  # TODO: handle exception
  def get_imu_data(self, joint_name:GloveIMUJointName) -> IMUData:
    return self.imu_data[joint_name.value]

  def set_imu_data(self, joint_name:GloveIMUJointName, data:IMUData) -> IMUData:
    self.imu_data[joint_name.value] = data

  def get_quaternion_data_numpy(self, joint_name:GloveQuaternionJointName=None) -> np.ndarray:
    if joint_name is None:
      return np.concatenate(list(map(lambda x:x.to_numpy(), self.quaternion_data)))
    return self.quaternion_data[joint_name.value].to_numpy()

  def get_quaternion_data(self, joint_name:GloveQuaternionJointName=None) -> list[QuaternionData]|QuaternionData:
    if joint_name is None:
      return self.quaternion_data
    return self.quaternion_data[joint_name.value]

  def set_quaternion_data(self, joint_name:GloveQuaternionJointName, data:QuaternionData) -> QuaternionData:
    self.quaternion_data[joint_name.value] = data