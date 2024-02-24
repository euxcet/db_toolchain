from .torch_node import TorchNode
import torch
import numpy as np
from db_graph.framework.graph import Graph
from db_graph.data.imu_data import IMUData
from db_graph.utils.window import Window
from db_graph.utils.filter import OneEuroFilter
from ...model.imu_trajectory_model import TrajectoryLSTMModel

class TrajectoryDetector(TorchNode):

  INPUT_EDGE_IMU         = 'imu'
  INPUT_EDGE_TOUCH_STATE = 'touch_state'
  OUTPUT_EDGE_RESULT     = 'result'

  def __init__(
      self,
      name: str,
      graph: Graph,
      input_edges: dict[str, str],
      output_edges: dict[str, str],
      checkpoint_file: str,
      imu_window_length: int,
      execute_interval: int,
      timestamp_step: float,
      move_threshold: float,
  ) -> None:
    super(TrajectoryDetector, self).__init__(
      name=name,
      graph=graph,
      input_edges=input_edges,
      output_edges=output_edges,
      model=TrajectoryLSTMModel(),
      checkpoint_file=checkpoint_file
    )
    self.touching = False
    self.imu_window_length = imu_window_length
    self.timestamp_step = timestamp_step
    self.move_threshold = move_threshold
    self.imu_window = Window[IMUData](imu_window_length)
    self.one_euro_filter = OneEuroFilter(np.zeros(2), 0)
    self.stable_window = Window(5)
    self.last_unstable_move = Window(30)
    self.counter.execute_interval = execute_interval

  def handle_input_edge_imu(self, data: IMUData, timestamp: float) -> None:
    self.imu_window.push(data)
    if self.counter.count() and self.imu_window.full() and self.touching:
      input_tensor = torch.tensor(self.imu_window.to_numpy_float().reshape(1, self.imu_window_length, 6)).to(self.device)
      output = self.model(input_tensor).detach().cpu().numpy().flatten()
      self.stable_window.push(np.linalg.norm(output) < self.move_threshold)
      if self.stable_window.last():
        output = np.zeros_like(output)
      if self.stable_window.full() and self.stable_window.all():
        self.last_unstable_move.clear()
      move = self.one_euro_filter(output, dt=self.timestamp_step)
      self.last_unstable_move.push(move)
      self.output(self.OUTPUT_EDGE_RESULT, move)

  def handle_input_edge_touch_state(self, data: str, timestamp: float) -> None:
    if data == 'touch_down':
      self.touching = True
    elif data in ['touch_up', 'click', 'double_click']:
      self.touching = False
      self.one_euro_filter = OneEuroFilter(np.zeros(2), 0)
      if self.last_unstable_move.capacity() > 0:
        self.output(self.OUTPUT_EDGE_RESULT, -self.last_unstable_move.sum())
        self.last_unstable_move.clear()
