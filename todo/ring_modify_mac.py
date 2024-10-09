import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../..'))

import time
import asyncio
import argparse
from ring import ring_pool, RingConfig

def connect(ring):
  asyncio.run(ring.connect())

def modify_mac(address, new_address, new_name):
  ring = ring_pool.add_ring(RingConfig(address=address, enable_imu=False, enable_touch=False), wait_until_initialized=True)
  ring.send_action('LEDSET=[R]')
  confirm = input('Modify mac to [{}] and name to [{}]? [yes/no]'.format(new_address, new_name))
  if confirm.strip().lower() in ['yes', 'y']:
    print(confirm)
    ring.send_action('DEVINFO')
    print('1')
    time.sleep(2)
    ring.send_action('DEVINFO={},{}'.format(new_address, new_name))
    print('1')
    time.sleep(2)
    ring.send_action('DEVINFO')
    print('1')
    time.sleep(2)
    ring.send_action('REBOOT')
    print('1')
    time.sleep(2)
    ring.send_action('disconnect')
    print('1')
  time.sleep(2)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--ring', type=str, required=True)
  parser.add_argument('--new-address', type=str, required=True)
  parser.add_argument('--new-name', type=str, required=True)
  args = parser.parse_args()
  modify_mac(args.ring, args.new_address, args.new_name)

# python ring_modify_mac.py --ring 0D85597D-C82C-E839-E0E0-4776246A6398 --new-address BB:AA:AA:AA:AA:AA --new-name "QHDX Ring X"
