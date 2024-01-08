from utils.window import Window
from demo.detector import Detector
from sensor.data import IMUData

class IMUStableDetector(Detector):
  def __init__(self, name:str, input_streams:dict[str, str], output_streams:dict[str, str],
               imu_window_length:int, execute_interval:int, threshold:float) -> None:
    super(IMUStableDetector, self).__init__(name=name, input_streams=input_streams, output_streams=output_streams)
    self.imu_window_length = imu_window_length
    self.stable_window = Window[IMUData](self.imu_window_length)
    self.threshold = threshold
    self.counter.execute_interval = execute_interval

  def handle_input_stream_imu(self, data:IMUData, timestamp:float) -> None:
    self.stable_window.push(data.acc_norm() < self.threshold)
    if self.counter.count(enable_print=True, print_fps=True) and self.stable_window.full():
      self.output(self.OUTPUT_STREAM_RESULT, self.stable_window.all())