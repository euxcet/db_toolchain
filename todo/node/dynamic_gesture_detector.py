import time
import torch
import torch.nn.functional as F
from ..model.imu_gesture_model import GestureNetCNN
from ..data.imu_data import IMUData
from ..framework.torch_node import TorchNode
from ..utils.window import Window

class DynamicGestureDetector(TorchNode):

  INPUT_EDGE_IMU     = 'imu'
  OUTPUT_EDGE_RESULT = 'result'

  def __init__(
      self,
      name:str,
      input_edges:dict[str, str],
      output_edges:dict[str, str],
      num_classes:int,
      imu_window_length:int,
      result_window_length:int,
      execute_interval:int,
      checkpoint_file:str,
      labels:list[str],
      confidence_threshold:list[float],
      trigger_wait_time:list[float],
      min_trigger_interval: list[float],
      block:dict,
  ) -> None:
    super(DynamicGestureDetector, self).__init__(
      name=name,
      input_edges=input_edges,
      output_edges=output_edges,
      model=GestureNetCNN(num_classes=num_classes),
      checkpoint_file=checkpoint_file
    )
    self.imu_window_length = imu_window_length
    self.num_classes = num_classes
    self.labels = labels
    self.confidence_threshold = confidence_threshold
    self.trigger_wait_time = trigger_wait_time
    self.min_trigger_interval = min_trigger_interval
    self.imu_window = Window[IMUData](self.imu_window_length)
    self.result_window = Window(result_window_length)
    self.last_gesture_time = [0] * num_classes
    self.trigger_time = [0] * num_classes
    self.block_until_time = [0] * num_classes
    self.block = block
    self.counter.print_interval = 12000
    self.counter.execute_interval = execute_interval

  def trigger(self, gesture_id:int, current_time:float) -> None:
    label:list[str] = self.labels
    # trigger priority
    if gesture_id is not None:
      if current_time > self.block_until_time[gesture_id]:
        self.trigger_time[gesture_id] = current_time
      block = self.block.get(label[gesture_id])
      if block is not None:
        block_gestures, block_times = block['gesture'], block['time']
        for block_gesture, block_time in zip(block_gestures, block_times):
          block_gesture_id = label.index(block_gesture)
          self.block_until_time[block_gesture_id] = current_time + block_time
          self.trigger_time[block_gesture_id] = 0

    # delayed trigger
    for i in range(self.num_classes):
      if self.trigger_time[i] > 0 and \
        current_time >= self.trigger_time[i] + self.trigger_wait_time[i]:
        self.trigger_time[i] = 0
        self.output(self.OUTPUT_EDGE_RESULT, label[i])
    
  def handle_input_edge_imu(self, data:IMUData, timestamp:float) -> None:
    self.imu_window.push(data.to_numpy())
    if self.counter.count(enable_print=True, print_fps=True) and self.imu_window.full():
      current_time = time.time()
      input_tensor = torch.tensor(self.imu_window.to_numpy_float().T
                                  .reshape(1, 6, 1, self.imu_window_length)).to(self.device)
      output_tensor = F.softmax(self.model(input_tensor).detach().cpu(), dim=1)
      gesture_id = torch.max(output_tensor, dim=1)[1].item()
      confidence = output_tensor[0][gesture_id].item()
      if gesture_id < len(self.confidence_threshold) and confidence > self.confidence_threshold[gesture_id]:
        self.result_window.push(gesture_id)
        if current_time > self.last_gesture_time[gesture_id] + self.min_trigger_interval[gesture_id] and \
          self.result_window.full() and self.result_window.all(lambda x: x == gesture_id):
           self.trigger(gesture_id, current_time)
           self.last_gesture_time[gesture_id] = current_time
      else:
        self.result_window.push(-1)
        self.trigger(None, current_time)
