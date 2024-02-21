from bleak import BleakClient
import asyncio

class NotifyProtocol():
  READ_CHARACTERISTIC  = 'BAE80011-4F05-4503-8E65-3AF1F7329D1F'
  WRITE_CHARACTERISTIC = 'BAE80010-4F05-4503-8E65-3AF1F7329D1F'

  GET_SOFTWARE_VERSION = bytearray([0x00, 0x00, 0x11, 0x00])
  GET_HARDWARE_VERSION = bytearray([0x00, 0x00, 0x11, 0x01])
  GET_BATTERY_LEVEL    = bytearray([0x00, 0x00, 0x12, 0x00])
  GET_BATTERY_STATUS   = bytearray([0x00, 0x00, 0x12, 0x01])

  GET_IMU = bytearray([0x00, 0x00, 0x31, 0x00, 0x00, 0x32, 0x01, 0x00, 0x00])


class RingV2():
  def __init__(self, address):
    self.address = address

  def connect(self) -> None:
    asyncio.run(self.connect_async())

  def on_disconnect(self, device) -> None:
    print('Disconnected')

  def notify_callback(self, sender, data:bytearray) -> None:
    print(len(data))
    for i in range(len(data)):
      print(data[i], end = ' ')
    print(data)
    # result = NotifyProtocol.decode_notify_callback(data)
    # if result[0] == NotifyProtocol.EDPT_QUERY_SS:
    #   self.output(self.OUTPUT_EDGE_BATTERY, result[1])
    # if result[0] == NotifyProtocol.EDPT_OP_TOUCH_ACTION:
    #   op_type, report_method, action_code = result[1:]
    #   if op_type < 2:
    #     self.log_info('Touch action method: ' + ['HID', 'BLE', 'HID & BLE'][report_method])
    #   elif op_type == 2:
    #     self.output(self.OUTPUT_EDGE_TOUCH, action_code)

  
  async def write(self, data: bytearray) -> None:
    await self.client.write_gatt_char(NotifyProtocol.WRITE_CHARACTERISTIC, data)

  async def connect_async(self) -> None:
    # async with BleakClient(self.address, disconnected_callback=self.on_disconnect) as self.client:
    #   await self.client.connect()
    #   services = await self.client.get_services()
    #   for service in services:
    #     print(f'Service: {service}')
    #     for char in service.characteristics:
    #       print(f'Characteristic: {char}')

    async with BleakClient(self.address, disconnected_callback=self.on_disconnect) as self.client:
      await self.client.connect()
      await self.client.start_notify(NotifyProtocol.READ_CHARACTERISTIC, self.notify_callback)
      await asyncio.sleep(1)
      await self.write(NotifyProtocol.GET_IMU)
      await asyncio.sleep(0.1)
      # await self.client.write_gatt_char(NotifyProtocol.WRITE_CHARACTERISTIC, bytearray([0x00, 0x00, 0x11, 0x1]))
      # await asyncio.sleep(0.1)
      while True:
        await asyncio.sleep(30)


    # await self.client.start_notify(NotifyProtocol.SPP_READ_CHARACTERISTIC, self.spp_notify_callback)
    # await self.client.start_notify(NotifyProtocol.NOTIFY_CHARACTERISTIC, self.notify_callback)
    # await self.client.write_gatt_char(NotifyProtocol.NOTIFY_CHARACTERISTIC, NotifyProtocol.do_op_touch_action(1, 1, 0))
    # await self.spp_write('ENSPP')
    # await self.spp_write('ENFAST')
    # await self.spp_write('TPOPS=' + '1,1,1' if self.enable_touch else '0,0,0')
    # if self.enable_imu:
    #   await self.spp_write('IMUARG=0,0,0,' + str(self.imu_freq))
    #   await self.spp_write('ENDB6AX')
    # await self.set_led_color(self.led_color)
    # self.on_connect()

    # while self.client.is_connected:
    #   await self._perform_action()
    #   await asyncio.sleep(0.2)
  


if __name__ == '__main__':
  RingV2('FD872584-6F9B-2FA7-F9E7-4E8B873EC273').connect()