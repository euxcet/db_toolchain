from ..utils.window import Window
from ..data.imu_data import IMUData
from ..framework.node import Node

class IMUStableDetector(Node):

  INPUT_EDGE_IMU     = 'imu'
  OUTPUT_EDGE_RESULT = 'result'

  def __init__(
      self,
      name:str, 
      input_edges:dict[str, str],
      output_edges:dict[str, str],
      imu_window_length:int,
      execute_interval:int,
      threshold:float,
  ) -> None:
    super(IMUStableDetector, self).__init__(
      name=name,
      input_edges=input_edges,
      output_edges=output_edges
    )
    self.name = name
    self.imu_window_length = imu_window_length
    self.stable_window = Window[IMUData](self.imu_window_length)
    self.threshold = threshold
    self.counter.execute_interval = execute_interval

  def handle_input_edge_imu(self, data:IMUData, timestamp:float) -> None:
    self.stable_window.push(data.acc_norm() < self.threshold)
    if self.counter.count(enable_print=True, print_fps=True) and self.stable_window.full():
      self.output(self.OUTPUT_EDGE_RESULT, self.stable_window.all())