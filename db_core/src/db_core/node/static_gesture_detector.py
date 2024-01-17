import time
import torch
import numpy as np
import torch.nn.functional as F
from ..utils.window import Window
from ..model.quat_gesture_model import FullyConnectedModel
from ..framework.torch_node import TorchNode

class StaticGestureDetector(TorchNode):

  INPUT_EDGE_QUATERNION = 'quaternion'
  OUTPUT_EDGE_RESULT    = 'result'

  def __init__(
      self,
      name:str,
      input_streams:dict[str, str],
      output_streams:dict[str, str],
      num_classes:int,
      execute_interval:int,
      result_window_length:int,
      checkpoint_file:str,
      labels:list[str],
      confidence_threshold:list[float],
      min_trigger_interval: list[float],
  ) -> None:
    super(StaticGestureDetector, self).__init__(
      name=name,
      input_streams=input_streams,
      output_streams=output_streams,
      model=FullyConnectedModel(num_classes=num_classes),
      checkpoint_file=checkpoint_file
    )
    self.num_classes = num_classes
    self.labels = labels
    self.confidence_threshold = confidence_threshold
    self.min_trigger_interval = min_trigger_interval
    self.result_window = Window(result_window_length)
    self.last_gesture_time = [0] * num_classes
    self.trigger_time = [0] * num_classes
    self.block_until_time = [0] * num_classes
    self.counter.print_interval = 1000
    self.counter.execute_interval = execute_interval
    
  def handle_input_stream_quaternion(self, data:np.ndarray, timestamp:float) -> None:
    if self.counter.count(enable_print=False, print_fps=True):
      current_time = time.time()
      input_tensor = torch.tensor(data.astype(np.float32)).reshape(1, 64).to(self.device)
      output_tensor = F.softmax(self.model(input_tensor).detach().cpu(), dim=1)
      gesture_id = torch.max(output_tensor, dim=1)[1].item()
      confidence = output_tensor[0][gesture_id].item()
      if gesture_id < len(self.confidence_threshold) and confidence > self.confidence_threshold[gesture_id]:
        self.result_window.push(gesture_id)
        if current_time > self.last_gesture_time[gesture_id] + self.min_trigger_interval[gesture_id] and \
          self.result_window.full() and self.result_window.all(lambda x: x == gesture_id):
          self.last_gesture_time[gesture_id] = current_time
          self.output(self.OUTPUT_EDGE_RESULT, self.labels[gesture_id])
      else:
        self.result_window.push(-1)