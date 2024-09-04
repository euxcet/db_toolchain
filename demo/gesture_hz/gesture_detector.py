import time
import torch
import torch.nn.functional as F
from db_graph.framework.graph import Graph
from db_graph.data.imu_data import IMUData
from db_graph.utils.window import Window
from db_zoo.node.algorithm.torch_node import TorchNode
from db_zoo.model.imu_inception_model import InceptionTimeModel

class GestureDetector(TorchNode):

  INPUT_EDGE_IMU     = 'imu'
  OUTPUT_EDGE_GESTURE = 'gesture'

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

    self.pinch_down = False
    self.tap_counter = 0

  def handle_input_edge_imu(self, data:IMUData, timestamp:float) -> None:
    self.imu_window.push(data.to_numpy())
    print(self.pinch_down)
    if self.counter.count(enable_print=False, print_fps=True) and self.imu_window.full():
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
