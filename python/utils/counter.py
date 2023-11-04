import time

class Counter():
  def __init__(self, print_gap:int=1000):
    self.t0 = 0
    self.counter = 0
    self.print_gap = print_gap

  def count(self, print_dict:dict=dict(), disable_print=False):
    current_time = time.time()
    if self.t0 == 0:
      self.t0 = current_time
    else:
      self.counter += 1
      if self.counter == self.print_gap:
        fps = self.counter / (current_time - self.t0)
        print_dict['FPS'] = fps
        if not disable_print:
          for key, value in print_dict.items():
            print('{}: {}'.format(key, value), end='  ')
          print()
        self.counter = 0
        self.t0 = 0