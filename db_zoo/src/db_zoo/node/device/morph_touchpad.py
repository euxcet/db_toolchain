import time
import numpy as np
from db_graph.data.morph_touchpad_data import MorphTouchpadData, MorphContactData
from db_graph.framework.device import Device, DeviceLifeCircleEvent
from db_graph.framework.graph import Graph
from .sensel import sensel

class MorphTouchpad(Device):

  OUTPUT_EDGE_LIFECYCLE = 'lifecycle'
  OUTPUT_EDGE_DATA = 'data'

  MAX_X = 230.0
  MAX_Y = 130.0
  FPS = 50

  def __init__(
      self,
      name: str,
      graph: Graph,
      input_edges: dict[str, str],
      output_edges: dict[str, str],
  ) -> None:
    super(MorphTouchpad, self).__init__(
      name=name,
      graph=graph,
      input_edges=input_edges,
      output_edges=output_edges,
    )

  # lifecycle callbacks
  def on_pair(self) -> None:
    self.log_info('Connecting')
    self.lifecycle_status = DeviceLifeCircleEvent.on_pair
    self.output(self.OUTPUT_EDGE_LIFECYCLE, DeviceLifeCircleEvent.on_pair)

  def on_connect(self) -> None:
    self.log_info("Connected")
    self.lifecycle_status = DeviceLifeCircleEvent.on_connect
    self.output(self.OUTPUT_EDGE_LIFECYCLE, DeviceLifeCircleEvent.on_connect)

  def on_disconnect(self) -> None:
    self.log_info("Disconnected")
    self.lifecycle_status = DeviceLifeCircleEvent.on_disconnect
    self.output(self.OUTPUT_EDGE_LIFECYCLE, DeviceLifeCircleEvent.on_disconnect)

  def on_error(self) -> None:
    self.lifecycle_status = DeviceLifeCircleEvent.on_error
    self.output(self.OUTPUT_EDGE_LIFECYCLE, DeviceLifeCircleEvent.on_error)

  def connect(self) -> None:
    self.on_pair()
    if (device_list := sensel.getDeviceList()[1]).num_devices > 0:
      self.handle = sensel.openDeviceByID(device_list.devices[0].idx)[1]
      self.info = sensel.getSensorInfo(self.handle)[1]
      sensel.setFrameContent(self.handle, 0x0F)
      sensel.setContactsMask(self.handle, 0x0F)
      self.frame = sensel.allocateFrameData(self.handle)[1]
      sensel.startScanning(self.handle)
      self.updated = False
      self.last_frame = None
    else:
      self.log_warning('Device not found.')
      return
    self.on_connect()

    self.is_running = True
    while self.is_running:
      sensel.readSensor(self.handle)
      num_frames = sensel.getNumAvailableFrames(self.handle)[1]
      for i in range(num_frames):
        while self.last_frame != None and (time.perf_counter() - self.last_frame.timestamp) * self.FPS < 1:
          pass
        sensel.getFrame(self.handle, self.frame)
      row = self.info.num_rows
      col = self.info.num_cols
      force_array = np.zeros((row, col))
      for r in range(row):
        force_array[r, :] = self.frame.force_array[r * col : (r + 1) * col]
      force_array *= 0.2
      frame = MorphTouchpadData(force_array, time.perf_counter())

      for i in range(self.frame.n_contacts):
        c = self.frame.contacts[i]
        contact = MorphContactData(
           c.id, c.state,
           c.x_pos / self.MAX_X, c.y_pos / self.MAX_Y,
           c.area, c.total_force, c.major_axis,
           c.minor_axis, c.delta_x, c.delta_y,
           c.delta_force, c.delta_area,
        )
        frame.append_contact(contact)

      self.last_frame = frame
      self.updated = True
      self.output(self.OUTPUT_EDGE_DATA, frame)
    self.disconnect()

  def disconnect(self) -> None:
    self.is_running = False
    sensel.freeFrameData(self.handle, self.frame)
    sensel.stopScanning(self.handle)
    sensel.close(self.handle)
    self.on_disconnect()

  def reconnect(self) -> None: ...
