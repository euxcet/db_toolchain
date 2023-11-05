import time
import asyncio
from core.ble_ring import RingBLE, scan_rings

async def connect_ring():
  scan_macs = await scan_rings()
  print(scan_macs)

if __name__ == '__main__':
  asyncio.run(connect_ring())
