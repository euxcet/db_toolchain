import time
import os.path as osp
from utils.file_utils import make_dir

class Timestamp():
  def __init__(self):
    self.start_timestamp = 0
    self.end_timestamp = 0
    self.save_file = None
    self.save_dir = None

  def get_save_file(self, round_id):
    return osp.join(self.save_dir, str(round_id) + '_timestamp.txt')

  def flush(self):
    if self.save_file is not None:
      self.save_file.flush()

  # override
  def set_save_dir(self, save_dir):
    self.save_dir = save_dir

  # override
  def start(self):
    pass

  # override
  def stop(self):
    pass

  # override
  def begin_round(self, round_id):
    make_dir(self.save_dir)
    self.flush()
    self.save_file = open(self.get_save_file(round_id), 'w')

  # override
  def end_round(self, round_id):
    self.flush()

  # override
  def begin_sample(self, sample_id):
    self.start_timestamp = time.time()

  # override
  def end_sample(self, sample_id):
    self.end_timestamp = time.time()
    self.save_file.write(str(self.start_timestamp) + " " + str(self.end_timestamp) + '\n')

  # override
  def exist_save_file(self, round_id):
    return osp.exists(self.get_save_file(round_id))
