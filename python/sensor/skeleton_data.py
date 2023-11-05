from enum import Enum

# TODO: visualize skeleton

class SkeletonType(Enum):
  quaternion = 0
  coordinate = 1

class SkeletonQuaternion():
  def __init__(self, qw:float, qx:float, qy:float, qz:float):
    self.qw = qw
    self.qx = qx
    self.qy = qy
    self.qz = qz
  
class SkeletonData():
  def __init__(self, skeleton_type:SkeletonType, raw_data:list[float]):
    self.skeleton_type = skeleton_type
    self.raw_data = raw_data
    self.data = [SkeletonQuaternion(*raw_data[i:i + 4]) for i in range(0, len(raw_data), 4)]