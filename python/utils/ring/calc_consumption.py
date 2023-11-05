import time
import asyncio
from core.ble_ring import RingBLE, scan_rings

last_check_time = 0
first_battery = -1
first_battery_time = -1

def battery_callback(battery):
  global first_battery, first_battery_time
  if first_battery == -1:
    first_battery = battery
    first_battery_time = time.time()
  else:
    if battery > 0:
      current_time = time.time()
      print('Battery(%):', battery,
            'Time(s):', int(current_time - first_battery_time),
            'Cost(%/s):', (first_battery - battery) / (time.time() - first_battery_time), flush=True)

async def connect_ring():
  global ring
  scan_macs = await scan_rings()
  ring = RingBLE(scan_macs[0], index=0, battery_callback=battery_callback)
  coroutines = [ring.connect()]
  await asyncio.gather(*coroutines)

if __name__ == '__main__':
  asyncio.run(connect_ring())