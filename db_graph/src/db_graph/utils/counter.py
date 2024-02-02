import time
from .logger import logger

class Counter():
  def __init__(self, execute_interval=10, print_interval:int=100):
    self.st = 0
    self.counter = 0
    self.execute_interval = execute_interval
    self.print_interval = print_interval

  def count(self, print_dict:dict=dict(), enable_print=False, print_fps=True) -> bool:
    current_time = time.time()
    self.st = current_time if self.st == 0 else self.st
    self.counter += 1
    if self.counter % self.print_interval == 0:
      if enable_print:
        if print_fps:
          print_dict['FPS'] = self.print_interval / (current_time - self.st)
        logger.info('  '.join(f'{k}: {v}' for k, v in print_dict.items()))
      self.st = current_time
    return self.counter % self.execute_interval == 0