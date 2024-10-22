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
from db_zoo.model.word_resnet_model import ResNet, TCNLSTM
from ahrs.filters import Madgwick
from enum import Enum, auto
from madgwick_helper import MadgwickHelper
from post_to_google_ime import get_character
from pynput import mouse

class State(Enum):
    STATE_UP = auto()  # 在空中
    STATE_DOWN = auto()  # 在地面

class Automaton:
    def __init__(self):
        self.state = State.STATE_UP

    def transition(self, event):
        if self.state == State.STATE_UP:
            if event == 4:  # 收到了按下
                self.state = State.STATE_DOWN
            elif event == 3:  # 收到了抬起
                self.state = State.STATE_UP
        elif self.state == State.STATE_DOWN:
            if event == 4:
                self.state = State.STATE_DOWN
            elif event == 3:
                self.state = State.STATE_UP

    def get_state(self):
        return self.state

    def get_output_event(self, event):
        output_event = 0
        if event == 3:  # 抬起
            if self.state == State.STATE_DOWN:
                output_event = 3
        elif event == 4:  # 按下
            if self.state == State.STATE_UP:
                output_event = 4
        return output_event

class GestureDetector(TorchNode):

  INPUT_EDGE_IMU      = 'imu'
  INPUT_EDGE_TOUCH    = 'touch'
  INPUT_EDGE_BATTERY    = 'battery'
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
      model= ResNet([2, 2, 2], num_classes=4),
      checkpoint_file=checkpoint_file,
    )
    self.move_model = TCNLSTM(
        input_size=6,
        hidden_size=128,
        output_size=2,
        num_channels=[64, 128],
        kernel_size=3,
        dropout=0.1,
    ) 
    self.move_model.load_state_dict(torch.load('checkpoint/move.pt'))
    self.move_model.eval()

    self.imu_window_length = imu_window_length
    self.num_classes = num_classes
    self.labels = labels
    self.confidence_threshold = confidence_threshold
    self.imu_window = Window[IMUData](self.imu_window_length)
    self.move_imu_window = Window[IMUData](13)
    self.imu_x_window = Window(50)
    self.gesture_window = Window(gesture_window_length)
    self.last_gesture_time = [0] * num_classes
    self.trigger_time = [0] * num_classes
    self.counter.print_interval = 400
    self.counter.execute_interval = execute_interval
    self.tap_counter = 0
    self.automaton = Automaton()
    self.mouse_controller = mouse.Controller()
    self.hx = (torch.zeros(1, 1, 128), torch.zeros(1, 1, 128))
    self.madgwick_helper = MadgwickHelper()
    self.last_state = State.STATE_UP
    self.position_x = 0
    self.position_y = 0
    self.trajectory = []

  def handle_input_edge_battery(self, battery: int, timestamp: float) -> None:
    ...

  def handle_input_edge_touch(self, data: int|str, timestamp: float) -> None:
    ...

  def detect_up(self):
      def is_stable(x: np.ndarray):
          return max(x) - min(x) < 3
      
      if (
          self.imu_x_window.full()
          and is_stable(self.imu_x_window.window[-10:])
          and is_stable(self.imu_x_window.window[:10])
          and np.mean(self.imu_x_window.window[-10:])
          - np.mean(self.imu_x_window.window[:10])
          > 4
      ):
          return 3
      return 0

  def handle_input_edge_imu(self, data: IMUData, timestamp: float) -> None:
    data.gyr_x -= -0.04
    data.gyr_y -= 0
    data.gyr_z -= 0.04

    self.imu_window.push(data.to_numpy())
    self.imu_x_window.push(data.to_numpy()[0])

    self.madgwick_helper.update(self.imu_window.last())
    without_gravity = self.madgwick_helper.data_without_gravity
    self.move_imu_window.push(np.array(without_gravity))

    if self.counter.count(enable_print=True, print_fps=True) and self.imu_window.full():
      input_tensor = torch.tensor(self.imu_window.to_numpy_float().T
                                  .reshape(1, 6, self.imu_window_length)).to(self.device)
      output_tensor = F.softmax(self.model(input_tensor).detach().cpu(), dim=1)
      gesture_id = torch.max(output_tensor, dim=1)[1].item()
      confidence = output_tensor[0][gesture_id].item()

      # print(gesture_id, self.labels[gesture_id])
      if confidence > self.confidence_threshold:
        if gesture_id != 3:
           gesture = self.detect_up()
        gesture = self.automaton.get_output_event(gesture_id + 1)
        self.automaton.transition(gesture_id + 1)
        if gesture in [3, 4]:
          for i in range(2):
              self.imu_window.push(self.imu_window.last())
          self.tap_counter += 1
      
      current_state = self.automaton.get_state()
      if current_state == State.STATE_DOWN and self.last_state == State.STATE_UP: # DOWN
        print('DOWN')
      elif current_state == State.STATE_UP and self.last_state == State.STATE_DOWN: # UP
        print('UP')
        get_character(self.trajectory)
        self.trajectory = []
        self.position_x, self.position_y = 0, 0
      if current_state == State.STATE_DOWN: # MOVE
        input_tensor = torch.tensor(self.move_imu_window.to_numpy_float()
                            .reshape(1, 13, 6)).to(self.device)
        output_tensor, self.hx = self.move_model(input_tensor, self.hx)
        output_tensor = output_tensor.detach().cpu().numpy()


        dx, dy = output_tensor[0][0][0], output_tensor[0][0][1]
        if data.gyr_norm < 0.1:
          dx, dy = 0, 0

        self.position_x += dx
        self.position_y += dy
        
        self.trajectory.append((self.position_x, self.position_y, int(time.time() * 1e6)))
        # self.mouse_controller.move(dx, dy)
      self.last_state = current_state