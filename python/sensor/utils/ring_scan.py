import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../..'))

import asyncio
from sensor.ring_ble import scan_rings

async def connect_ring():
  scan_macs = await scan_rings()
  print(scan_macs)

if __name__ == '__main__':
  asyncio.run(connect_ring())
