import asyncio
from bleak import BleakClient, BleakScanner

async def scan():
  devices = await BleakScanner.discover()
  for device in devices:
    if device.name is not None and (device.name.startswith('BCL') or 'Ring' in device.name):
      print(device.address)
      print(device.details)
      

if __name__ == '__main__':
  asyncio.run(scan())