from concurrent.futures import ThreadPoolExecutor
from .device import Device

class DeviceManager():
  def __init__(self):
    self.devices: dict[str, Device] = {}
    self.executor = ThreadPoolExecutor(max_workers=20, thread_name_prefix='Device')
  
  def add_device(self, device: Device):
    self.devices[device.name] = device

  def start(self):
    for device in self.devices.values():
      self.executor.submit(device.connect)

device_manager = DeviceManager()