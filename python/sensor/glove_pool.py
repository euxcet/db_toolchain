from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from utils.logger import logger
from sensor.glove import Glove, GloveConfig

class GlovePool():
  def __init__(self, keep_alive=True):
    self.keep_alive = keep_alive
    # key is the ip address of the glove
    self.gloves:dict[tuple[str, int], Glove] = {}
    self.executor = ThreadPoolExecutor(max_workers=20, thread_name_prefix='Glove')
    if self.keep_alive:
      keep_alive_thread = Thread(target=self.keep_alive_strategy)
      keep_alive_thread.daemon = True
      keep_alive_thread.start()

  def keep_alive_strategy(self): ...

  def add_device(self, config:GloveConfig) -> Glove:
    if config.address in self.gloves:
      logger.warning(f'Glove[{config.ip}:{config.port}] is already added.')
      return
    glove = Glove(config)
    self.gloves[config.address] = glove
    self.executor.submit(glove.connect)
    return glove

  def get_device(self, config:GloveConfig, connect_when_miss:bool=False) -> Glove:
    if config.address in self.gloves:
      return self.gloves[config.address]
    if connect_when_miss:
      return self.add_device(config)
    logger.warning(f'The required glove[{config.ip}:{config.port}] has not been added yet, \
                     please consider using the connect_when_miss parameter.')
    return None

glove_pool = GlovePool(keep_alive=True)