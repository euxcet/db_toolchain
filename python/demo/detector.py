from __future__ import annotations
import torch
import torch.nn as nn
from utils.counter import Counter
from sensor import Ring, RingEvent, ring_pool
from sensor.glove import Glove, GloveEvent, glove_pool
from utils.register import Register

class DetectorEvent():
  def __init__(self, detector:str, data):
    self.detector = detector
    self.data = data

class DetectorEventBroadcaster():
  def __init__(self):
    self.detectors:list[Detector] = []

  def add_detector(self, detector:str):
    self.detectors.append(detector)

  def remove_detector(self, detector:str):
    self.detectors.append(detector)

  def broadcast_event(self,event):
    for d in self.detectors:
      d.handle_detector_event(event)

broadcaster = DetectorEventBroadcaster()

class Detector():
  # TODO: use a thread to handle event?
  def __init__(self, name:str, model:nn.Module=None, device:Ring|Glove=None, checkpoint_file:str=None, handler=None):
    self.name = name
    self.counter = Counter()
    self.handler = handler
    # model
    if model is not None:
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      self.model:nn.Module = model
      self.model.load_state_dict(torch.load(checkpoint_file, map_location=self.device))
      self.model.eval()
    # device
    if device is not None:
      if type(device) is Ring:
        ring_pool.bind_ring(self.handle_ring_event, ring=device)
      elif type(device) is Glove:
        glove_pool.bind_glove(self.handle_glove_event, glove=device)
    broadcaster.add_detector(self)

  def broadcast_event(self, data):
    event = DetectorEvent(self.name, data)
    if self.handler is not None:
      self.handler(event)
    broadcaster.broadcast_event(event)

  def handle_ring_event(self, device, event:RingEvent):
    pass

  def handle_glove_event(self, device, event:GloveEvent):
    pass

  def handle_detector_event(self, event:DetectorEvent):
    pass

  def __init_subclass__(cls) -> None:
    detector_register.register(cls.__name__, cls)

detector_register = Register[Detector]()