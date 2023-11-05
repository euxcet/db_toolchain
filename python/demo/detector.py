from __future__ import annotations
import torch
import torch.nn as nn
from abc import abstractmethod
from utils.counter import Counter

class DetectorEventBroadcaster():
  def __init__(self):
    self.detectors:list[Detector] = []

  def add_detector(self, detector:Detector):
    self.detectors.append(detector)

  def remove_detector(self, detector:Detector):
    self.detectors.append(detector)

  def broadcast_event(self, detector:Detector, event):
    for d in self.detectors:
      d.handle_detector_event(detector, event)

broadcaster = DetectorEventBroadcaster()

class Detector():
  def __init__(self, model:nn.Module=None, checkpoint_path=None, handler=None):
    if model is not None:
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      self.checkpoint_path = checkpoint_path
      self.model:nn.Module = model
      self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
      self.model.eval()
    self.counter = Counter()
    self.handler = handler
    broadcaster.add_detector(self)

  def broadcast_event(self, event):
    if self.handler is not None:
      self.handler(self, event)
    broadcaster.broadcast_event(self, event)

  @abstractmethod
  def run():
    pass

  @abstractmethod
  def handle_detector_event(self, detector, event):
    pass

  # @abstractmethod
  # def event_description(self) -> str:
  #   pass
