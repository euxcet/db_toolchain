import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))

import time
import argparse
import itertools
import os.path as osp
from pynput import keyboard
from record.device import get_devices
from utils.file_utils import load_json

class Recorder():
  def __init__(self, args, config):
    self.config = config
    self.user = args.user

    self.categories = config['categories']
    self.category = self.get_category(args.category)

    self.samples_per_round = config['samples_per_round']
    self.save_dir = config['save_dir']
    self.sample_time = config['sample_time']
    self.restore_time = config['restore_time']

    self.listener = keyboard.Listener(on_press=self.on_press)
    print(self.config['devices'])
    self.devices = get_devices(self.config['devices'])
    self.set_device_save_dir()
    print('User: {}   Category: {}'.format(self.user, self.category))
    self.running = True

  # life cycle
  def start(self):
    self.running = True
    self.listener.start()
    for device in self.devices:
      device.start()

  def stop(self):
    self.running = False
    self.listener.stop()
    for device in self.devices:
      device.stop()

  def begin_round(self, round_id):
    for device in self.devices:
      device.begin_round(round_id)
    print(f'[Begin Round {round_id}]')

  def end_round(self, round_id):
    for device in self.devices:
      device.end_round(round_id)
    print(f'[End Round {round_id}]')

  def begin_sample(self, sample_id):
    # highlighting only works in unix
    print('\033[0;32;40m' + f'[Begin Sample {sample_id + 1}/{self.sample_number}]' + '\033[0m')
    for device in self.devices:
      device.begin_sample(sample_id)

  def end_sample(self, sample_id):
    print('\033[0;31;40m' + f'[End Sample {sample_id + 1}/{self.sample_number}]' + '\033[0m')
    for device in self.devices:
      device.end_sample(sample_id)

  def get_round_id(self):
    if len(self.devices) == 0:
      return 0
    for round_id in itertools.count():
      if any([not device.exist_save_file(round_id) for device in self.devices]):
        return round_id

  def record(self):
    round_id = self.get_round_id()
    self.begin_round(round_id)
    for sample_id in range(self.sample_number):
      time.sleep(self.restore_time)
      self.begin_sample(sample_id)
      time.sleep(self.sample_time)
      self.end_sample(sample_id)
    self.end_round(round_id)

  # handle keyboard events
  def on_press(self, key):
    functions = {
      '.': self.stop,
      'c': self.switch_category,
      'u': self.switch_user,
      'n': self.switch_number,
      keyboard.Key.space: self.record,
    }
    if hasattr(key, 'char'):
      if key.char in functions:
        functions[key.char]()
    else:
      if key in functions:
        functions[key]()

  def switch_category(self):
    category = input('Input the category id or category name (type int/string):')
    self.category = self.get_category(category)
    self.set_device_save_dir()
    print(f'OK,', self.category)

  def switch_user(self):
    user = input('Input the user name (type string):')
    self.user = user
    self.set_device_save_dir()
    print(f'OK,', self.user)

  def switch_number(self):
    number = input('Input the number of samples in each round (type int):')
    assert number.isdigit()
    self.sample_number = int(number)
    print(f'OK', self.sample_number)

  def set_device_save_dir(self):
    for device in self.devices:
      device.set_save_dir(osp.join(self.save_dir, self.user, self.category))

  def get_category(self, category:str):
    # TODO: handle exception
    if category in self.categories:
      return category
    elif category.isdigit():
      return self.categories[int(category)]
    raise NotImplementedError()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--user', type=str, required=True)
  parser.add_argument('--category', type=str, required=True)
  parser.add_argument('--config', type=str, required=True)
  args = parser.parse_args()

  recorder = Recorder(args, load_json(args.config))
  recorder.start()

  while recorder.running:
    time.sleep(1)
