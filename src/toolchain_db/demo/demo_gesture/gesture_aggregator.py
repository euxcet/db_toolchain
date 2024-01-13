import time
import numpy as np
from demo.detector import Detector
from utils.logger import logger

from threading import Thread
import playsound

class GestureAggregator(Detector):
  def __init__(self, name:str, input_streams:dict[str, str], output_streams:dict[str, str], gestures:list, play_sound=True) -> None:
    super(GestureAggregator, self).__init__(name=name, input_streams=input_streams, output_streams=output_streams)
    self.gestures = [self.Gesture(**gesture_config) for gesture_config in gestures]
    self.play_sound = play_sound

  def play(self, name) -> None:
    playsound.playsound('sound/' + name + '.mp3')

  def handle_input_stream_event(self, event:str, timestamp:float) -> None:
    for gesture in self.gestures:
      if gesture.update(event):
        logger.info(f'Trigger [{gesture.name}]')
        if self.play_sound:
          Thread(target=self.play, args=(gesture.name,)).start()

  class Gesture():
    def __init__(self, name, events:list[str], window_time:float=1.0, min_trigger_interval:float=1.0, keep_order=False) -> None:
      self.name = name
      self.events = events
      self.window_time = window_time
      self.min_trigger_interval = min_trigger_interval
      self.events_timestamp = np.zeros(len(events))
      self.keep_order = keep_order
      self.last_trigger_time = 0

    def update(self, event:str) -> bool:
      if event not in self.events:
        return False
      current_time = time.time()
      self.events_timestamp[self.events.index(event)] = current_time
      if np.max(self.events_timestamp) - np.min(self.events_timestamp) < self.window_time \
         and (not self.keep_order or np.all(self.events_timestamp[:-1] <= self.events_timestamp[1:])) \
         and current_time > self.last_trigger_time + self.min_trigger_interval:
         self.last_trigger_time = current_time
         return True
      return False
