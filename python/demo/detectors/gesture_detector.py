import time
import torch
import numpy as np
import torch.nn.functional as F
from utils.window import Window
from model.imu_gesture_model import GestureNetCNN
from demo.detector import Detector, DetectorEvent
from sensor.imu_data import IMUData
from sensor import Ring, RingEvent, RingEventType
from sensor.glove import Glove, GloveEvent, GloveEventType

class GestureDetector(Detector):
  def __init__(self, device:Ring|Glove=None, handler=None, arguments:dict=dict()):
    super(GestureDetector, self).__init__(
      model=GestureNetCNN(num_classes=arguments['num_classes']), device=device,
      handler=handler, arguments=arguments)
    self.imu_window_length = self.arguments['imu_window_length']
    self.imu_window = Window[IMUData](self.imu_window_length)
    self.result_window = Window(self.arguments['result_window_length'])
    self.counter.execute_interval = self.arguments['execute_interval']
    self.last_gesture_time = [0] * arguments['num_classes']
    self.trigger_time = [0] * arguments['num_classes']
    self.block_until_time = [0] * arguments['num_classes']
    self.threshold = self.arguments['confidence_threshold']

  def trigger(self, gesture_id:int, current_time:float):
    label:list[str] = self.arguments['label']
    # trigger priority
    if gesture_id is not None:
      if current_time > self.block_until_time[gesture_id]:
        self.trigger_time[gesture_id] = current_time
      block = self.arguments['block'].get(label[gesture_id])
      if block is not None:
        block_gestures, block_times = block['gesture'], block['time']
        for block_gesture, block_time in zip(block_gestures, block_times):
          block_gesture_id = label.index(block_gesture)
          self.block_until_time[block_gesture_id] = current_time + block_time
          self.trigger_time[block_gesture_id] = 0

    # delayed trigger
    for i in range(self.arguments['num_classes']):
      if self.trigger_time[i] > 0 and \
        current_time >= self.trigger_time[i] + self.arguments['trigger_wait_time'][i]:
        self.trigger_time[i] = 0
        self.broadcast_event(label[i])
    
  def detect(self, data):
    self.imu_window.push(data)
    if self.counter.count() and self.imu_window.full():
      current_time = time.time()
      input_tensor = torch.tensor(self.imu_window.to_numpy_float().T
                                  .reshape(1, 6, 1, self.imu_window_length)).to(self.device)
      output_tensor = F.softmax(self.model(input_tensor).detach().cpu(), dim=1)
      gesture_id = torch.max(output_tensor, dim=1)[1].item()
      confidence = output_tensor[0][gesture_id].item()
      if gesture_id < len(self.threshold) and confidence > self.threshold[gesture_id]:
        self.result_window.push(gesture_id)
        if current_time > self.last_gesture_time[gesture_id] + 0.8 and self.result_window.full() and \
           self.result_window.all(lambda x: x == gesture_id):
           self.trigger(gesture_id, current_time)
           self.last_gesture_time[gesture_id] = current_time
      else:
        self.result_window.push(-1)
        self.trigger(None, current_time)

  def handle_detector_event(self, event:DetectorEvent):
    pass

  def handle_ring_event(self, device, event:RingEvent):
    if event.event_type == RingEventType.imu:
      self.detect(event.data.to_numpy())

  def handle_glove_event(self, device, event:GloveEvent):
    if event.event_type == GloveEventType.imu_6axis:
      gyr_x, gyr_y, gyr_z, acc_x, acc_y, acc_z = event.data[3]
      self.detect(np.array([acc_x, acc_y, acc_z, -9.8 * gyr_x, -9.8 * gyr_y, -9.8 * gyr_z]))