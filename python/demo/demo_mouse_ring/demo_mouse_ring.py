# class Ring():
#   def __init__(self, ring, input_data_queue:queue.Queue):
#     self.ring = BLERing(ring, index=0, imu_callback=self.imu_callback)
#     self.ring_thread = Thread(target=self.connect)
#     self.ring_thread.daemon = True
#     self.ring_thread.start()
#     self.input_data_queue = input_data_queue

#   def connect(self):
#     asyncio.run(self.ring.connect())

#   def imu_callback(self, index:int, data:IMUData):
#     self.input_data_queue.put(data)


# class GestureDetector():
#   GESTURE_IMU_LENGTH = 200
#   RESULT_WINDOW_LENGTH = 3
#   CALCULATE_STEP = 6
#   CONFIDENCE_THRESHOLD = 0.9
#   def __init__(self, model_path, handler):
#     self.model = GestureNetRing(num_classes=10).to('cpu')
#     self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
#     self.model.eval()
#     self.handler = handler
#     self.counter = Counter()
#     self.gesture_name = {
#       0: 'touch_down',
#       1: 'touch_up',
#       3: 'click',
#       4: 'double_click',
#     }
#     self.imu_data = Window(self.GESTURE_IMU_LENGTH)
#     self.gesture_window = Window(self.RESULT_WINDOW_LENGTH)

#     self.trigger_time = {self.gesture_name[x]: None for x in self.gesture_name}
#     self.block_time = {self.gesture_name[x]: 0 for x in self.gesture_name}

#   def filter(self, gesture=None):
#     current_time = time.time()
#     for t_gesture in self.trigger_time:
#       if self.trigger_time[t_gesture] is not None and current_time > self.trigger_time[t_gesture] + 0.3:
#         if current_time > self.block_time[t_gesture] + 1:
#           self.handler(t_gesture)
#         self.trigger_time[t_gesture] = None

#     if gesture is not None:
#       if gesture == 'touch_up' or gesture == 'double_click' or gesture == 'touch_down':
#         if current_time > self.block_time[gesture] + 1:
#           self.handler(gesture)
#       if gesture == 'double_click':
#         self.block_time['touch_down'] = current_time
#         self.block_time['click'] = current_time
#         self.trigger_time['click'] = None
#       if gesture == 'click':
#         self.block_time['touch_down'] = current_time
#         self.trigger_time['click'] = current_time

#   def update(self, data:IMUData):
#     self.imu_data.push(data)
#     self.counter.count(disable_print=True)
#     last_gesture_time = {x: 0 for x in self.gesture_name}
#     if self.imu_data.full() and self.counter.counter % self.CALCULATE_STEP == 0:
#       self.filter()
#       data = self.imu_data.map(lambda x: x.to_numpy()).to_numpy().reshape(-1, 6).T
#       input_tensor = torch.from_numpy(data).float().view(1, 6, 1, self.GESTURE_IMU_LENGTH).to('cpu')
#       outputs = F.softmax(self.model(input_tensor).cpu().detach(), dim=1)
#       _, predictions = torch.max(outputs, 1)
#       gesture, confidence = predictions[0].item(), outputs[0][predictions[0].item()].item()
#       if gesture in self.gesture_name and confidence > self.CONFIDENCE_THRESHOLD:
#         self.gesture_window.push(gesture)
#         current_time = time.time()
#         if current_time > last_gesture_time[gesture] + 1 and self.gesture_window.full() and \
#             self.gesture_window.count(lambda x:x == self.gesture_window.first()) == self.gesture_window.capacity():
#           self.filter(self.gesture_name[gesture])
#           last_gesture_time[gesture] = current_time
#       else:
#         self.gesture_window.push(-1)

# class RingMouse():
#   TRAJECTORY_SCALE = 10
#   def __init__(self, ring:str, touch_model:str, trajectory_model:str):
#     self.input_data_queue = queue.Queue()
#     self.ring = Ring(ring, self.input_data_queue)
#     self.mouse = Controller()
#     self.trajectory_detector = TrajectoryDetector(trajectory_model, self.trajectory_handler)
#     self.gesture_detector = GestureDetector(touch_model, self.gesture_handler)
#     self.start_position = (0, 0)
#     self.current_move = (0, 0)
#     self.reset()

#   def length(self, x):
#     return math.sqrt(x[0] * x[0] + x[1] * x[1])

#   def reset(self):
#     self.mouse.position = (330, 330 + 66)

#   def trajectory_handler(self, dx, dy):
#     if self.length((dx, dy)) < 0.1:
#       self.current_move = (0, 0)
#     else:
#       self.current_move = (self.current_move[0] + dx, self.current_move[1] + dy)
#     self.mouse.move(dx * self.TRAJECTORY_SCALE, dy * self.TRAJECTORY_SCALE)

#   def gesture_handler(self, gesture):
#     if gesture == 'touch_down':
#       self.start_position = self.mouse.position
#       self.trajectory_detector.set_touching(True)
#     elif gesture == 'touch_up':
#       self.mouse.move(-self.current_move[0] * self.TRAJECTORY_SCALE, -self.current_move[1] * self.TRAJECTORY_SCALE)
#       self.current_move = (0, 0)
#       self.trajectory_detector.set_touching(False)
#     else:
#       if gesture == 'click':
#         if self.trajectory_detector.touching:
#           self.trajectory_detector.set_touching(False)
#           self.mouse.position = self.start_position
#         self.mouse.click(Button.left)
#       elif gesture == 'double_click':
#         self.reset()

#   def run(self):
#     while True:
#       data = self.input_data_queue.get()
#       self.trajectory_detector.update(data)
#       self.gesture_detector.update(data)

import argparse
from sensor import ring_pool, RingConfig
from .detector.trajectory_detector import TrajectoryDetector

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-r', '--ring', type=str, default='043C2EAC-6125-DEA0-7011-CF8D0D7DBEBA')
  parser.add_argument('-tc', '--touch_checkpoint', type=str, default='touch_all.pth')
  parser.add_argument('-jc', '--trajectory_checkpoint', type=str, default='trajectory_all_1101.pth')
  args = parser.parse_args()

  ring_pool.add_ring(RingConfig(args.ring, 'MouseRing'))
  trajectory_detector = TrajectoryDetector(args.trajectory_model, None)
