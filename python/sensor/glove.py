import queue
import socket
import struct
from utils.logger import logger
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

class Glove():
  def __init__(self):
    pass

# class GloveConfig():
#   def __init__(self, ip:str='127.0.0.1'):
#     self.ip = ip

# class Glove():
#   def __init__(self, config:GloveConfig):
#     self.first_timestamp = 0
#     self.count = 0
#     self.save_file_name = None

#   def parse_pose_from_data(self, data):
#     qw, = struct.unpack('f', data[0:4])
#     qx, = struct.unpack('f', data[4:8])
#     qy, = struct.unpack('f', data[8:12])
#     qz, = struct.unpack('f', data[12:16])
#     return Pose(qw, qx, qy, qz)

#   def process_data(self, data):
#     if data.decode('cp437').find('VRTRIX') == 0:
#       radioStrength, = struct.unpack('h', data[265:267])
#       battery, = struct.unpack('f', data[267:271])
#       calScore, = struct.unpack('h', data[271:273])
#       for i in range(16):
        
#       joint_quat = []
#       for i in range(0, 16):
#         joint_quat.append(self.parse_pose_from_data(data[9 + 16 * i : 25 + 16 * i])) 

#       current_time = time.time()
#       if self.first_timestamp == 0:
#         self.first_timestamp = current_time
#       else:
#         self.count += 1
#         if current_time > self.first_timestamp + 10:
#           print('RadioStrength: {}, Battery: {}, CalScore: {}, FPS: {:.2f}'.format(
#             -radioStrength, battery, calScore, self.count / (current_time - self.first_timestamp)))
#           self.first_timestamp = 0
#           self.count = 0

#       if self.count % 10 == 0:
#         input_data = torch.tensor(np.array([[j.qw, j.qx, j.qy, j.qz] for j in joint_quat]).astype(np.float32).reshape(1, 64))
#         output = self.model(input_data).cpu().detach().numpy().flatten()
#         gesture_id = np.argmax(output)
#         print(gesture_id, output[gesture_id])


#   def run(self):
#     sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     server_address = (self.ip, 11002)
#     print('Connecting to %s port %s' % server_address)
#     sock.connect(server_address)
#     print('Glove connected.')

#     client_sock = None

#     while self.streaming:
#       data = sock.recv(1024)
#       if client_sock is not None:
#         client_sock.send(data)
#       self.process_data(data)


class GlovePool():
  def __init__(self, keep_alive=True):
    self.keep_alive = keep_alive
    # key is the ip address of the glove
    self.gloves:dict[tuple[str, str], Glove] = {}
    self.handlers:dict[str, set] = {}
    self.executor = ThreadPoolExecutor(max_workers=20, thread_name_prefix='Glove')
    self.event_queue:queue.Queue[GloveEvent] = queue.Queue()
    event_distribution_thread = Thread(target=self.event_handler)
    event_distribution_thread.daemon = True
    event_distribution_thread.start()
    if self.keep_alive:
      keep_alive_thread = Thread(target=self.keep_alive_strategy)
      keep_alive_thread.daemon = True
      keep_alive_thread.start()

  def keep_alive_strategy(self):
    pass

  def add_ring(self, config:RingConfig) -> Ring:
    if config.address in self.rings:
      logger.warning(f'Ring[{config.address}] is already added.')
      return
    ring = Ring(config, event_queue=self.event_queue)
    self.rings[config.address] = ring
    self.handlers[config.address] = set()
    self.executor.submit(ring.run)
    return ring

  def get_ring(self, config:RingConfig, connect_when_miss:bool=None) -> Ring:
    if config.address in self.rings:
      return self.rings[config.address]
    if connect_when_miss:
      return self.add_ring(config)
    logger.warning(f'The required ring[{config.address}] has not been added yet, \
                     please consider using the connect_when_miss parameter.')
    return None

  def bind_ring(self, event_handler, ring:Ring=None, config:RingConfig=None, address:str=None):
    ring_address = None if address is None else address
    ring_address = ring_address if config is None else config.address
    ring_address = ring_address if ring is None else ring.address
    if ring_address is None:
      logger.error(f'The address for the ring is not provided.')
      raise Exception()
    if ring_address not in self.rings:
      logger.error(f'Ring[{ring_address}] is not in the ring set.')
      raise Exception()
    self.handlers[ring_address].add(event_handler)

  def event_handler(self):
    while True:
      event = self.event_queue.get()
      for handler in self.handlers[event.address]:
        handler(self.rings[event.address], event)

ring_pool = RingPool(keep_alive=True)