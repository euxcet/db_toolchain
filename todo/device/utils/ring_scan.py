import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../..'))

import asyncio
from bleak import BleakScanner

async def scan_rings():
  ring_macs = []
  devices = await BleakScanner.discover()
  for d in devices:
    if d.name is not None and 'ring' in d.name.lower():
      ring_macs.append(d.address)
  return ring_macs

if __name__ == '__main__':
  asyncio.run(scan_rings())
