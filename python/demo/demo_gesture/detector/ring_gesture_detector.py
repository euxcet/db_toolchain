import time
import torch
import torch.nn.functional as F
from threading import Thread
from queue import Queue
import torch
from utils.counter import Counter
from threading import Thread
from queue import Queue
from ...core.window import Window
from ...model.gesture_cnn import GestureNetCNN

class RingGestureDetector(Thread):
  def __init__(self, arguments, event_handler):
    Thread.__init__(self)
    self.arguments = arguments
    self.event_handler = event_handler
    self.load_model(self.arguments)
    self.counter = Counter()
    self.data_queue = Queue()
    # Ring
    self.move_name = {
      0: 'left',
      1: 'forward',
      2: 'up',
    }
    self.gesture_name = {
      3: 'bomb',
      4: 'close',
      5: 'forward',
      6: 'gather',
      7: 'quick',
      8: 'enemy',
      9: 'cover',
      10: 'repeat',
    }

  def load_model(self, arguments):
    self.device = 'cuda' if torch.cuda.is_available else 'cpu'
    self.model = GestureNetCNN(num_classes=arguments['num_classes']).to(self.device)
    self.model.load_state_dict(torch.load(arguments['model_file']))
    self.model.eval()

  def run(self):
    imu_data = Window(200)
    gesture_window = Window(6)
    move_window = Window(6)
    last_gesture_time = 0
    last_move_time = 0
    while True:
      data = self.data_queue.get()
      imu_data.push(data)
      self.counter.count()
      if imu_data.full() and self.counter.counter % 10 == 0:
        data = imu_data.map(lambda x: x.to_numpy()).to_numpy().reshape(-1, 6).T
        input_tensor = torch.from_numpy(data).float().view(1, 6, 1, 200).to('cpu')
        outputs = F.softmax(self.model(input_tensor).cpu().detach(), dim=1)
        _, predictions = torch.max(outputs, 1)
        gesture, confidence = predictions[0].item(), outputs[0][predictions[0].item()].item()
        if gesture in self.gesture_name and confidence > 0.95:
          gesture_window.push(gesture)
          current_time = time.time()
          if current_time > last_gesture_time + 1.2 and gesture_window.full() and gesture_window.count(lambda x:x == gesture_window.first()) == gesture_window.capacity():
            self.event_handler(self.gesture_name[gesture])
            last_gesture_time = current_time
        else:
          gesture_window.push(-1)

        if gesture in self.move_name and confidence > 0.95:
          move_window.push(gesture)
          current_time = time.time()
          if current_time > last_move_time + 1.2 and move_window.full() and move_window.count(lambda x:x == move_window.first()) == move_window.capacity():
            self.event_handler("move")
            last_move_time = current_time
        else:
          move_window.push(-1)

