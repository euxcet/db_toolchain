import time
from typing import Callable
from stream.stream import Stream
from threading import Thread

class StreamManager():
  def __init__(self) -> None:
    self.streams:dict[str, Stream] = {}
    self.stream_handlers:dict[str, list[Callable]] = {}
    self.indirect_stream_count = 0
    Thread(target=self.distribute).start()

  def get_stream(self, name:str) -> Stream:
    return self.streams[name]

  def add_stream(self, stream:Stream) -> Stream:
    if not stream.direct:
      self.indirect_stream_count += 1
    self.streams[stream.name] = stream
    self.stream_handlers[stream.name] = []
    return stream

  def bind_stream(self, name:str, handler:Callable) -> None:
    if self.streams[name].direct:
      self.streams[name].bind(handler)
    else:
      self.stream_handlers[name].append(handler)

  def unbind_stream(self, name:str, handler:Callable) -> None:
    if self.streams[name].direct:
      self.streams[name].unbind(handler)
    else:
      self.stream_handlers[name].remove(handler)

  def distribute(self):
    while True:
      if self.indirect_stream_count > 0:
        for name, stream in self.streams.items():
          data = stream.get_no_wait()
          if data is not None:
            for handler in self.stream_handlers[name]:
              handler(*data)
        time.sleep(0.001)
      else:
        time.sleep(0.01)

stream_manager = StreamManager()