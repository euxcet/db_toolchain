import time
import copy
import torch
from collections import deque
import torch.nn.functional as F
import numpy as np
import pyquaternion as pyq
from db_graph.framework.graph import Graph
from db_graph.data.imu_data import IMUData
from db_graph.utils.window import Window
from db_zoo.node.algorithm.torch_node import TorchNode
from db_zoo.model.imu_inception_model import InceptionTimeModel
from ahrs.filters import Madgwick
from utils import ButterBandpassRealTimeFilter
from pynput import mouse

class GestureDetector(TorchNode):

  INPUT_EDGE_IMU      = 'imu'
  INPUT_EDGE_TOUCH    = 'touch'
  OUTPUT_EDGE_GESTURE = 'gesture'
  OUTPUT_EDGE_ORIENTATION = 'orientation'

  def __init__(
      self,
      name: str,
      graph: Graph,
      input_edges: dict[str, str],
      output_edges: dict[str, str],
      num_classes: int,
      imu_window_length: int,
      gesture_window_length: int,
      execute_interval: int,
      checkpoint_file: str,
      confidence_threshold: float,
      labels: list[str],
  ) -> None:
    super(GestureDetector, self).__init__(
      name=name,
      graph=graph,
      input_edges=input_edges,
      output_edges=output_edges,
      model=InceptionTimeModel(input_channels=6, num_classes=num_classes),
      checkpoint_file=checkpoint_file,
    )
    self.imu_window_length = imu_window_length
    self.num_classes = num_classes
    self.labels = labels
    self.confidence_threshold = confidence_threshold
    self.imu_window = Window[IMUData](self.imu_window_length)
    self.gesture_window = Window(gesture_window_length)
    self.last_gesture_time = [0] * num_classes
    self.trigger_time = [0] * num_classes
    self.counter.print_interval = 400
    self.counter.execute_interval = execute_interval
    self.is_moving = True
    self.pinch_down = False
    self.tap_counter = 0
    self.madgwick_filter = Madgwick(Dt=1 / 200)
    self.orientation_queue = deque[pyq.Quaternion](maxlen=2)
    self.last_orientation = pyq.Quaternion(0.012, -0.825, 0.538, 0.174)
    self.last_position = np.array([0, 0, 0])
    self.butter_filter_x = ButterBandpassRealTimeFilter(200)
    self.butter_filter_y = ButterBandpassRealTimeFilter(200)
    self.mouse_controller = mouse.Controller()

  def update_orientation(self, data: IMUData) -> None:
    orientation = pyq.Quaternion(
      self.madgwick_filter.updateIMU(
        self.last_orientation.elements,
        gyr=[data.gyr_x, data.gyr_y, data.gyr_z],
        acc=[data.acc_x, data.acc_y, data.acc_z],
      )
    )
    self.output(self.OUTPUT_EDGE_ORIENTATION, orientation)
    self.last_orientation = orientation

    position = np.array(orientation.rotate([1, 0, 0]))
    self.last_position = position

    self.orientation_queue.append(copy.deepcopy(orientation))
    if len(self.orientation_queue) < 2:
        return

    # if deque is full, calculate delta orientation and move cursor
    delta_orientation = self.orientation_queue[0].inverse * orientation
    delta_orientation = np.array(delta_orientation.yaw_pitch_roll)

    if self.is_moving:
        x = delta_orientation[0] * 1000
        y = delta_orientation[1] * 1000
        # if x * x + y * y > 8:
        #   self.mouse_controller.move(x, y)  # finger mode

    self.orientation_queue.clear()
    self.orientation_queue.append(copy.deepcopy(orientation))

  def handle_input_edge_touch(self, data: int, timestamp: float) -> None:
    if data == 9:
      self.is_moving = True
    elif data == 10:
      self.is_moving = False

  def handle_input_edge_imu(self, data: IMUData, timestamp: float) -> None:
    self.update_orientation(data)
    self.imu_window.push(data.to_numpy())
    if self.counter.count(enable_print=True, print_fps=True) and self.imu_window.full():
      input_tensor = torch.tensor(self.imu_window.to_numpy_float().T
                                  .reshape(1, 6, self.imu_window_length)).to(self.device)
      output_tensor = F.softmax(self.model(input_tensor).detach().cpu(), dim=1)
      gesture_id = torch.max(output_tensor, dim=1)[1].item()
      confidence = output_tensor[0][gesture_id].item()
      if confidence > self.confidence_threshold:
        self.gesture_window.push(gesture_id)
        if gesture_id != 0:
            print(self.labels[gesture_id])
        if self.labels[gesture_id] == 'pinch_down':
            self.pinch_down = True
        elif self.labels[gesture_id] in ['pinch', 'pinch_up']:
            self.pinch_down = False
        if self.labels[gesture_id] in ['pinch', 'middle_pinch', 'clap', 'snap', 'tap_plane', \
                                       'tap_air', 'circle_clockwise', 'circle_counterclockwise']:
            print(gesture_id, self.labels[gesture_id], self.tap_counter)
            for i in range(self.imu_window_length):
                self.imu_window.push(self.imu_window.last())
            self.tap_counter += 1
