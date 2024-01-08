from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from utils.counter import Counter
from utils.register import Register
from stream import Stream, stream_manager

class Detector():
  OUTPUT_STREAM_RESULT = "result"
  def __init__(self, name:str, input_streams:dict[str, str], output_streams:dict[str, str],
               model:nn.Module=None, checkpoint_file:str=None):
    self.name = name
    self.counter = Counter()
    # model
    if model is not None:
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      self.model:nn.Module = model
      self.model.load_state_dict(torch.load(checkpoint_file, map_location=self.device))
      self.model.eval()

    self.output_stream:dict[str, Stream] = {}
    # stream
    for stream, local_stream_name in output_streams.items():
      self.output_stream[stream] = stream_manager.add_stream(Stream(self.get_stream_name(local_stream_name)))
    for stream, stream_name in input_streams.items():
      if type(stream_name) is list:
        for name in stream_name:
          stream_manager.bind_stream(name, getattr(self, f'handle_input_stream_{stream}'))
      else:
        stream_manager.bind_stream(stream_name, getattr(self, f'handle_input_stream_{stream}'))

  def output(self, local_stream_name:str, data:Any):
    self.output_stream[local_stream_name].put(data)

  def get_stream_name(self, local_stream_name:str):
    return self.name + "_" + local_stream_name

  def __init_subclass__(cls) -> None:
    detector_register.register(cls.__name__, cls)

detector_register = Register[Detector]()