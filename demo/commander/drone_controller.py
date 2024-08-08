import time
import numpy as np
from enum import Enum
from typing_extensions import override
from db_graph.data.imu_data import IMUData
from db_graph.framework.graph import Graph
from db_graph.framework.node import Node
from db_graph.utils.window import Window
import pyquaternion as pyq
from ahrs.filters import Madgwick


class DroneMode(Enum):
  LAND = 0
  TAKEOFF = 1
  MOVE = 2
  TURN = 3
  TURNING = 4

class DroneAction(Enum):
  LAND = 0
  TAKEOFF = 1
  MOVE_FORWARD = 2
  MOVE_BACKWARD = 3
  STOP = 4
  TURN = 5
  TURN_LEFT = 6
  TURN_RIGHT = 7
  STOP_TURN = 8

class HandDirection(Enum):
  UP = 0
  DOWN = 1
  FORWARD = 2
  UNKNOWN = 3



"""
takeoff 下垂->抬起->下垂->抬起 z -1 -> 1 -> -1 -> 1
land 抬起->下垂->抬起->下垂 z 1 -> -1 -> 1 -> -1
move 抬起->水平->抬起->水平 speed x 
stop 抬起 stop
turn_left 抬起->水平->左转 rc 0 0 0 -20
turn_right 抬起->水平->右转 rc 0 0 0 20
"""

class DroneController(Node):

  INPUT_EDGE_IMU = 'imu'
  OUTPUT_EDGE_COMMAND = 'command'

  def __init__(
      self,
      name: str,
      graph: Graph,
      input_edges: dict[str, str],
      output_edges: dict[str, str],
  ) -> None:
    super().__init__(
      name=name,
      graph=graph,
      input_edges=input_edges,
      output_edges=output_edges,
    )
    self.madgwick_filter = Madgwick(Dt=1 / 200)
    self.orientation_queue = Window(5)
    self.last_orientation = pyq.Quaternion(0.012, -0.825, 0.538, 0.174)
    self.last_position = np.array([0, 0, 0])
    self.stable = Window(50)
    self.direction_window = Window(200)
    self.imu_window = Window[IMUData](200)
    self.mode = DroneMode.LAND
    self.wait_for_turn = False
    self.last_action = None
    self.last_action_timestamp = 0
    self.last_stable_position = None

  @override
  def start(self):
    ...

  def perform_action(self, action: DroneAction):
    timestamp = time.time()
    if timestamp - self.last_action_timestamp > 1.5 or \
       (timestamp - self.last_action_timestamp > 1 and action != self.last_action):
      self.last_action = action
      self.last_action_timestamp = timestamp
      if action == DroneAction.LAND:
        self.output(self.OUTPUT_EDGE_COMMAND, 'land')
        # refactor: use response
        self.mode = DroneMode.LAND
      elif action == DroneAction.TAKEOFF:
        self.output(self.OUTPUT_EDGE_COMMAND, 'takeoff')
        self.mode = DroneMode.TAKEOFF
      elif action == DroneAction.MOVE_FORWARD:
        self.output(self.OUTPUT_EDGE_COMMAND, 'rc 0 20 0 0')
        self.mode = DroneMode.MOVE
      elif action == DroneAction.MOVE_BACKWARD:
        self.output(self.OUTPUT_EDGE_COMMAND, 'rc 0 -20 0 0')
        self.mode = DroneMode.MOVE
      elif action == DroneAction.TURN:
        self.mode = DroneMode.TURN
      elif action == DroneAction.TURN_LEFT:
        self.output(self.OUTPUT_EDGE_COMMAND, 'rc 0 0 0 -20')
        self.mode = DroneMode.TURNING
      elif action == DroneAction.TURN_RIGHT:
        self.output(self.OUTPUT_EDGE_COMMAND, 'rc 0 0 0 20')
        self.mode = DroneMode.TURNING
      elif action == DroneAction.STOP_TURN:
        self.output(self.OUTPUT_EDGE_COMMAND, 'stop')
        self.mode = DroneMode.TURN
      elif action == DroneAction.STOP:
        self.output(self.OUTPUT_EDGE_COMMAND, 'stop')
        self.last_action_timestamp -= 0.5
        if self.mode != DroneMode.LAND:
          self.mode = DroneMode.TAKEOFF

  def check_direction_gesture(self, gesture: list[HandDirection]) -> bool:
    if not self.direction_window.full() or len(gesture) == 0:
      return False
    current = 0
    for direction in self.direction_window.window:
      if direction == gesture[current]:
        current += 1
        if current == len(gesture):
          return True
    return False

  def direction_gesture(self):
    if not self.direction_window.full():
      return
    if self.check_direction_gesture([HandDirection.DOWN, HandDirection.UP, HandDirection.DOWN, HandDirection.UP]):
      self.perform_action(DroneAction.TAKEOFF)
    elif self.check_direction_gesture([HandDirection.UP, HandDirection.DOWN, HandDirection.UP, HandDirection.DOWN]):
      self.perform_action(DroneAction.LAND)
    elif self.check_direction_gesture([HandDirection.UP, HandDirection.FORWARD, HandDirection.UP, HandDirection.FORWARD]) and \
         HandDirection.DOWN not in self.direction_window.window:
      x = 0
      y = 0
      z = 0
      for data in self.imu_window.window:
        x += data.acc_x
        y += data.acc_z
        z += data.acc_z
      if z < -900:
        self.perform_action(DroneAction.MOVE_FORWARD)
      elif z > 900:
        self.perform_action(DroneAction.MOVE_BACKWARD)
    elif self.check_direction_gesture([HandDirection.UP, HandDirection.FORWARD]):
      self.wait_for_turn = True

  def handle_input_edge_imu(self, data: IMUData, timestamp: float) -> None:
    self.imu_window.push(data)
    orientation = pyq.Quaternion(
      self.madgwick_filter.updateIMU(
        self.last_orientation.elements,
        gyr=[data.gyr_x, data.gyr_y, data.gyr_z],
        acc=[data.acc_x, data.acc_y, data.acc_z],
      )
    )
    position = np.array(orientation.rotate([1, 0, 0]))

    if position[2] < -0.85:
      self.direction_window.push(HandDirection.DOWN)
    elif position[2] > 0.85:
      self.direction_window.push(HandDirection.UP)
    elif position[2] > -0.3 and position[2] < 0.3:
      self.direction_window.push(HandDirection.FORWARD)
    else:
      self.direction_window.push(HandDirection.UNKNOWN)

    if position[2] < -0.6 or position[2] > 0.6:
      self.wait_for_turn = False

    self.direction_gesture()

    self.stable.push(np.linalg.norm(position - self.last_position) < 0.01)
    if self.stable.full() and self.stable.all():
      # stop
      if self.direction_window.last() == HandDirection.UP:
        self.perform_action(DroneAction.STOP)
      # turn
      if self.wait_for_turn and self.mode not in [DroneMode.TURN, DroneMode.TURNING] \
        and self.direction_window.last() == HandDirection.FORWARD:
        self.last_stable_position = position
        self.perform_action(DroneAction.TURN)
      # turn left and right
      if self.mode == DroneMode.TURN or self.mode == DroneMode.TURNING:
        dot = position[0] * self.last_stable_position[1] - position[1] * self.last_stable_position[0]
        if dot < -0.3 or dot > 0.3:
          if self.mode == DroneMode.TURNING:
            self.perform_action(DroneAction.STOP_TURN)
          else:
            if dot < 0:
              self.perform_action(DroneAction.TURN_LEFT)
            else:
              self.perform_action(DroneAction.TURN_RIGHT)
          self.last_stable_position = position

    self.last_orientation = orientation
    self.last_position = position
    self.orientation_queue.push(orientation)