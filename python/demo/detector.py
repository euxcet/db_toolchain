from __future__ import annotations
import torch
import torch.nn as nn
from abc import abstractmethod
from utils.counter import Counter
from sensor import Ring, ring_pool
from sensor.glove import Glove, glove_pool

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

  def broadcast_event(self, event):
    for d in self.detectors:
      d.handle_detector_event(event)

broadcaster = DetectorEventBroadcaster()

class Detector():
  def __init__(self, model:nn.Module=None, device:Ring|Glove=None, handler=None, arguments:dict=dict()):
    # model
    if model is not None:
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      self.model:nn.Module = model
      self.model.load_state_dict(torch.load(arguments['checkpoint_file'], map_location=self.device))
      self.model.eval()
    # device
    if device is not None:
      if type(device) is Ring:
        ring_pool.bind_ring(self.handle_ring_event, ring=device)
      elif type(device) is Glove:
        glove_pool.bind_glove(self.handle_glove_event, glove=device)
    self.counter = Counter()
    self.handler = handler
    self.arguments = arguments
    broadcaster.add_detector(self)

  def broadcast_event(self, data):
    event = DetectorEvent(self.name, data)
    if self.handler is not None:
      self.handler(event)
    broadcaster.broadcast_event(event)

  @property
  @abstractmethod
  def name(self):
    return self.arguments['name']

  @abstractmethod
  def handle_detector_event(self, event:DetectorEvent):
    pass