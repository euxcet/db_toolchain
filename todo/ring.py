from bleak import BleakClient, BleakScanner
import time
import asyncio
import struct
import math

class NotifyProtocol():
  READ_CHARACTERISTIC  = 'BAE80011-4F05-4503-8E65-3AF1F7329D1F'
  WRITE_CHARACTERISTIC = 'BAE80010-4F05-4503-8E65-3AF1F7329D1F'

  GET_SOFTWARE_VERSION = bytearray([0x00, 0x00, 0x11, 0x00])
  GET_HARDWARE_VERSION = bytearray([0x00, 0x00, 0x11, 0x01])
  GET_BATTERY_LEVEL    = bytearray([0x00, 0x00, 0x12, 0x00])
  GET_BATTERY_STATUS   = bytearray([0x00, 0x00, 0x12, 0x01])
  OPEN_6AXIS_IMU = bytearray([0x00, 0x00, 0x40, 0x06])
  CLOSE_6AXIS_IMU = bytearray([0x00, 0x00, 0x40, 0x00])
  GET_TOUCH = bytearray([0x00, 0x00, 0x61, 0x00])
  OPEN_AUDIO = bytearray([0x00, 0x00, 0x71, 0x00, 0x01])
  CLOSE_AUDIO = bytearray([0x00, 0x00, 0x71, 0x00, 0x00])
  GET_NFC = bytearray([0x00, 0x00, 0x82, 0x00])

class RingV2():
  def __init__(self, address):
    self.address = address
    self.times = []

  def connect(self) -> None:
    asyncio.run(self.connect_async())

  def on_disconnect(self, device) -> None:
    print('Disconnected')

  def notify_callback(self, sender, data:bytearray) -> None:
    self.times.append(time.time())
    if len(self.times) > 20:
      self.times.pop(0)
    if len(self.times) > 10:
      print(data[5])
      print((len(self.times) - 1) / (self.times[-1] - self.times[0]) * 4)

    # if data[2] == 0x40 and data[3] == 0x06:
    #   if len(data) > 20:
    #     acc_x, acc_y, acc_z = struct.unpack('hhh', data[5:11])
    #     acc_x, acc_y, acc_z = acc_x / 1000 * 9.8, acc_y / 1000 * 9.8, acc_z / 1000 * 9.8
    #     gyr_x, gyr_y, gyr_z = struct.unpack('hhh', data[11:17])
    #     gyr_x, gyr_y, gyr_z = gyr_x / 180 * math.pi,  gyr_y / 180 * math.pi, gyr_z / 180 * math.pi
    #     print(round(acc_x, 2), round(acc_y, 2), round(acc_z, 2), round(gyr_x, 2), round(gyr_y, 2), round(gyr_z, 2))
    # elif data[2] == 0x61 and data[3] == 0x0:
    #   print("Touch", data[4])
    # elif data[2] == 0x61 and data[3] == 0x1:
    #   print("Touch Raw", data[4:])

  async def write(self, data: bytearray) -> None:
    await self.client.write_gatt_char(NotifyProtocol.WRITE_CHARACTERISTIC, data)

  async def connect_async(self) -> None:
    async with BleakClient(self.address, disconnected_callback=self.on_disconnect) as self.client:
      await self.client.connect()
      await self.client.start_notify(NotifyProtocol.READ_CHARACTERISTIC, self.notify_callback)
      await asyncio.sleep(1)
      await self.write(NotifyProtocol.OPEN_6AXIS_IMU)
      await asyncio.sleep(0.1)
      while True:
        await asyncio.sleep(30)

async def scan():
  devices = await BleakScanner.discover()
  for device in devices:
    print(device)

if __name__ == '__main__':
  # asyncio.run(scan())
  RingV2('F1E6125A-A60D-F925-5E73-0F1845082C36').connect()