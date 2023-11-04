import time
import argparse
import itertools
import os.path as osp
from pynput import keyboard
from record.device import get_devices

class Recorder():
  def __init__(self, args):
    self.user = args.user
    self.action = args.action
    self.sample_number = args.number

    # TODO query save_dir from the dataset manager.
    self.save_dir = args.save_dir

    self.listener = keyboard.Listener(on_press=self.on_press)
    # devices
    self.devices = get_devices(args.device)
    self.set_device_save_dir()
    
    self.running = True
    self.processing = False

    self.sample_time = args.sample_time
    self.restore_time = args.restore_time

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
    if self.processing:
      print('Ignore key')
      return
    self.processing = True
    if key == keyboard.Key.space:
      self.record()
    try:
      {'.': self.stop, 'a': self.switch_action, 'u': self.switch_user, 'n': self.switch_number}[key.char]()
    except:
      pass
    self.processing = False

  def switch_action(self):
    action = input('Input new action name:')
    print(f'Change action to {action}')
    self.action = action
    self.set_device_save_dir()

  def switch_user(self):
    user = input('Input new user name:')
    print(f'Change user to {user}')
    self.user = user
    self.set_device_save_dir()

  def switch_number(self):
    number = input('Input number of samples:')
    print(f'Change number of samples to {number}')
    self.sample_number = int(number)

  def set_device_save_dir(self):
    for device in self.devices:
      device.set_save_dir(osp.join(self.save_dir, self.user, self.action))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--name', type=str, required=True)
  parser.add_argument('--user', type=str, required=True)
  parser.add_argument('--action', type=str, required=True)
  parser.add_argument('--number', type=int, required=True, default=10)
  parser.add_argument('--device', type=str, required=True)
  parser.add_argument('--sample_time', type=float, required=False, default=1.0)
  parser.add_argument('--restore_time', type=float, required=False, default=1.0)
  args = parser.parse_args()
  # todo: help

  recorder = Recorder(args)
  recorder.start()

  while recorder.running:
    time.sleep(1)
