import cv2
import time
import numpy as np
import queue
import face_recognition
from threading import Thread
from ultralytics import YOLO
from typing_extensions import override
from db_graph.data.imu_data import IMUData
from db_graph.framework.graph import Graph
from db_graph.framework.node import Node
from db_graph.utils.counter import Counter

class FaceDetector(Node):

  INPUT_EDGE_VIDEO = 'video'
  OUTPUT_EDGE_FACES = 'faces'

  def __init__(
      self,
      name: str,
      graph: Graph,
      input_edges: dict[str, str],
      output_edges: dict[str, str],
  ) -> None:
    super().__init__(
      name=name,
      graph=graph,
      input_edges=input_edges,
      output_edges=output_edges,
    )
    self.frame_queue = queue.Queue()
    # self.predict_queue = queue.Queue()
    self.counter = Counter(print_interval=50)
    self.predict_thread = Thread(target=self.predict)

  @override
  def start(self):
    self.predict_thread.start()

  def predict(self) -> list:
    while True:
        frame = self.frame_queue.get()
        while not self.frame_queue.empty():
            frame = self.frame_queue.get()
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        self.output(self.OUTPUT_EDGE_FACES, (face_locations, face_encodings))

  @override
  def block(self):
    ...
    # while True:
    #     frame = self.frame_queue.get()
    #     self.counter.count(enable_print=True)
    #     self.predict_queue.put(frame)
    #     cv2.imshow('frame', frame)
    #     cv2.waitKey(3)

  def handle_input_edge_video(self, data, timestamp: float) -> None:
    self.frame_queue.put(data)