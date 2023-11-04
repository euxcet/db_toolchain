import socket
import struct
import asyncio
from core.imu_data import IMUData
from core.ble_ring import BLERing, scan_rings
class RingGroup:
  def __init__(self, macs:list[str]):
    self.macs = macs
    self.rings:list[BLERing] = []
    self.initialize_rings()
    sk = socket.socket()
    sk.bind(('127.0.0.1', 9999))
    sk.listen(3)
    self.socket, _ = sk.accept()

  def initialize_rings(self):
    for index, mac in enumerate(self.macs):
      self.rings.append(BLERing(mac, index=index, imu_callback=self.imu_callback))

  async def connect_rings(self):
    coroutines = [ring.connect() for ring in self.rings]
    await asyncio.gather(*coroutines)

  def imu_callback(self, index:int, data:IMUData):
    self.socket.send(bytearray([0x44, 0x55, 0x66]))
    self.socket.send(struct.pack('i', index))
    self.socket.send(struct.pack('f', data.acc_x))
    self.socket.send(struct.pack('f', data.acc_y))
    self.socket.send(struct.pack('f', data.acc_z))
    self.socket.send(struct.pack('f', data.gyr_x))
    self.socket.send(struct.pack('f', data.gyr_y))
    self.socket.send(struct.pack('f', data.gyr_z))

  def run(self):
    asyncio.run(self.connect_rings())

def get_data():
  macs = asyncio.run(scan_rings())
  ring_group = RingGroup([macs[0]])
  ring_group.run()

if __name__ == '__main__':
  get_data()
