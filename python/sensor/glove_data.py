from enum import Enum
# from .imu_data import IMUData

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

  def __init__(self):
    return self.value

class GloveIMUJointName(Enum):
  WRIST_JOINT = 0
  THUMB_INTERMEDIATE = 1
  THUMB_DISTAL = 2
  INDEX_INTERMEDIATE = 3
  INDEX_DISTAL = 4
  MIDDLE_INTERMEDIATE = 5
  MIDDLE_DISTAL = 6
  RING_INTERMEDIATE = 7
  RING_DISTAL = 8
  PINKY_INTERMEDIATE = 9
  PINKY_DISTAL = 10

  def __init__(self):
    return self.value

class GloveData():
  def __init__(self, ):
    pass

if __name__ == '__main__':
  print(GloveQuaternionJointName.MIDDLE_DISTAL)