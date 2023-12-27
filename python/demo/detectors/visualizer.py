import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal, QObject
import vispy.scene as scene
from vispy.scene import Line
from vispy.color import Color

import pyquaternion as pyq
import math
from enum import Enum

VALID_SEQUENCE = [ 0, 1, 2, 3, 6, 7, 8, 11, 12, 13, 16, 17, 18, 21, 22, 23, ]


class Hand_Type(Enum):
    BOTH_HAND = 0
    HAND_LEFT = 1
    HAND_RIGTH = 2

# fmt: off
indices = [0, 1, 1, 2, 2, 3, 3, 4, 0, 5, 5, 6, 6, 7, 7, 8, 8, 9, 0, 10,
           10, 11, 11, 12, 12, 13, 13, 14, 0, 15, 15, 16, 16, 17, 17, 18, 18, 19, 0, 20,
           20, 21, 21, 22, 22, 23, 23, 24, 2, 6, 6, 11, 11, 16, 16, 21, 25, 0, 25, 2,
           25, 6, 25, 11, 25, 16, 25, 21, 1, 5, 5, 10, 10, 15, 15, 20, 25, 1, 25, 5,
           25, 10, 25, 15, 25, 20]

colors = ['#faff72', '#fff143', '#fff143', '#ffb61e', '#ffb61e', '#96ce54', '#96ce54', '#f47983', '#faff72', '#fff143',
          '#fff143', '#ffb61e', '#ffb61e', '#96ce54', '#96ce54', '#f47983', '#f47983', '#f36838', '#faff72', '#fff143',
          '#fff143', '#ffb61e', '#ffb61e', '#96ce54', '#96ce54', '#f47983', '#f47983', '#f36838', '#faff72', '#fff143',
          '#fff143', '#ffb61e', '#ffb61e', '#96ce54', '#96ce54', '#f47983', '#f47983', '#f36838', '#faff72', '#fff143',
          '#fff143', '#ffb61e', '#ffb61e', '#96ce54', '#96ce54', '#f47983', '#f47983', '#f36838', '#ffb61e', '#ffb61e',
          '#ffb61e', '#ffb61e', '#ffb61e', '#ffb61e', '#ffb61e', '#ffb61e', '#eaff56', '#faff72', '#eaff56', '#ffb61e',
          '#eaff56', '#ffb61e', '#eaff56', '#ffb61e', '#eaff56', '#ffb61e', '#eaff56', '#ffb61e', '#fff143', '#fff143',
          '#fff143', '#fff143', '#fff143', '#fff143', '#fff143', '#fff143', '#eaff56', '#fff143', '#eaff56', '#fff143',
          '#eaff56', '#fff143', '#eaff56', '#fff143', '#eaff56', '#fff143']
face_colors = ['#eaff56', '#faff72', '#fff143', '#ffb61e', '#96ce54', '#f47983', '#fff143', '#ffb61e', '#96ce54',
               '#f47983', '#f36838', '#fff143', '#ffb61e', '#96ce54', '#f47983', '#f36838', '#fff143', '#ffb61e',
               '#96ce54', '#f47983', '#f36838', '#fff143', '#ffb61e', '#96ce54', '#f47983', '#f36838']
# fmt: on


def vrtrix_evaluate_all_posi(joints_quat):  # 16,4  2o 26,3
  joints_quat = joints_quat.tolist()
  edges = [(0, 1, 16, 0.2),   (5, 6, 17, 1.5),   (10, 11, 18, 1.5), (15, 16, 19, 1.5), (20, 21, 20, 1.5),
            (1, 2, 1, 0.7),    (2, 3, 2, 0.5),    (3, 4, 3, 0.2),    (6, 7, 4, 0.7),    (7, 8, 5, 0.5),
            (8, 9, 6, 0.2),    (11, 12, 7, 0.8),  (12, 13, 8, 0.5),  (13, 14, 9, 0.2),  (16, 17, 10, 0.7),
            (17, 18, 11, 0.5), (18, 19, 12, 0.2), (21, 22, 13, 0.6), (22, 23, 14, 0.4), (23, 24, 15, 0.2),]
  init_vector = np.array([1, 0, 0])
  for angle in [40, 15, 0, -15, -30]:
    joints_quat.append(joints_quat[0] * pyq.Quaternion(axis=[0, 1, 0], angle=angle / 180 * math.pi))
  skeleton = [np.zeros(3) for _ in range(26)]
  for edge in edges:
    skeleton[edge[1]] = skeleton[edge[0]] + pyq.Quaternion(joints_quat[edge[2]]).rotate(init_vector) * edge[3]
  skeleton = [x * 8 for x in skeleton]
  return skeleton

def singnal_glove_all_data_consumption(
    gesture_data: Gesture,
):  # 将(16,4)的四元数转换为(26,3)的xyz数据
    hand_type = Hand_Type(gesture_data.hand_id)
    for frame_glove_raw_data in gesture_data.raw_data:
        joints_quat = []  # 长为64的列表
        for i in VALID_SEQUENCE:
            joints_quat.append(frame_glove_raw_data[i].get("quat_w"))
            joints_quat.append(frame_glove_raw_data[i].get("quat_x"))
            joints_quat.append(frame_glove_raw_data[i].get("quat_y"))
            joints_quat.append(frame_glove_raw_data[i].get("quat_z"))
        assert len(joints_quat) == 64, "nums of valid joints is not 16"  # 16,4
        joints_quat = np.array(joints_quat).reshape(16, 4)
        # 先计算VRtrix16个关节点xyz
        global_posi = vrtrix_evaluate_all_posi(joints_quat)
        # 绘制手势
        return (hand_type, global_posi)


class Visualizer(QObject):
  TEXT_SIGNAL = pyqtSignal(object)
  GESTURE_SIGNAL = pyqtSignal(object)

  def __init__(self, gesture_vis_widget: QWidget) -> None:
    super().__init__()
    self.name = "gesture visualzer"
    self.gesture_vis_widget = gesture_vis_widget
    self._init_canvas()
    self.GESTURE_SINGNAL.connect(self._on_gesture_data)
    self.TEXT_SINGNAL.connect(self._on_intention_data)

  def _init_canvas(self):
    self.canvas = scene.SceneCanvas(keys="interactive", bgcolor="transparent")
    hex_color = "#141c26"
    rgba_color = Color(hex_color).rgba
    self.canvas.bgcolor = rgba_color
    self.gesture_vis_widget.setAttribute(Qt.WA_TranslucentBackground, True)
    self._init_view()
    layout = QVBoxLayout(self.gesture_vis_widget)
    layout.addWidget(self.canvas.native)

  def _init_view(self):
    self.view = self.canvas.central_widget.add_view()
    self.view.camera = scene.cameras.ArcballCamera(fov=0)
    self.view.camera.scale_factor = 50  # 相机缩放

    self.__hand_keys = np.random.normal(size=(26, 3), loc=0, scale=1)
    # Create and show visual
    self.markers = scene.visuals.Markers(
      pos=self.__hand_keys,
      antialias=False,
      face_color="red",
      edge_color="white",
      edge_width=0,
      scaling=True,
    )
    self.markers.parent = self.view.scene
    # 画线
    line_pos = np.array([self.__hand_keys[i] for i in indices])
    self.__line = Line(pos=line_pos, connect="segments", width=5, antialias=True)
    self.__line.parent = self.view.scene

  def _on_gesture_data(self, pb_data):
    hand_type = pb_data[0]
    gesture_data = pb_data[1]
    if gesture_data is not None:
      self.__hand_keys = self.__pos_fix(hand_type, gesture_data)
      self.markers.set_data( pos=self.__hand_keys, size=1.2, edge_width=0, face_color=face_colors)
      line_pos = np.array([self.__hand_keys[i] for i in indices])
      self.__line.set_data(pos=line_pos, color=colors, width=16)

  def __pos_fix( self, hand_type, target_array):
    if hand_type == Hand_Type.HAND_LEFT:
      position = np.array(target_array)
      position[:, [1, 2]] = position[:, [2, 1]]
      position[:, 0] = -position[:, 0]
      position[:, 1] = -position[:, 1]
      position[:, 2] = -position[:, 2]
      return position
