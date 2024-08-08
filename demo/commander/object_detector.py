import cv2
import time
import numpy as np
import queue
from threading import Thread
from ultralytics import YOLO
from typing_extensions import override
from db_graph.data.imu_data import IMUData
from db_graph.framework.graph import Graph
from db_graph.framework.node import Node
from db_graph.utils.counter import Counter

class ObjectDetector(Node):

  INPUT_EDGE_VIDEO = 'video'
  OUTPUT_EDGE_OBJECTS = 'objects'

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
    self.predict_queue = queue.Queue()
    self.model = YOLO("yolov8n.pt")
    self.boxes = None
    self.counter = Counter(print_interval=50)
    self.predict_thread = Thread(target=self.predict)
    self.predict_thread.start()

  @override
  def start(self):
    ...

  def predict(self) -> list:
    while True:
        frame = self.predict_queue.get()
        while not self.predict_queue.empty():
            frame = self.predict_queue.get()
        results = self.model.track(source=frame, persist=True, tracker='bytetrack.yaml')
        self.boxes = results[0].boxes
        self.output(self.OUTPUT_EDGE_OBJECTS, self.boxes)

  @override
  def block(self):
    while True:
        frame = self.frame_queue.get()
        self.counter.count(enable_print=True)
        self.predict_queue.put(frame)
        if self.boxes is not None:
            for box, conf in zip(self.boxes.xyxy, self.boxes.conf):
                if conf > 0.5:
                    frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
        cv2.imshow('frame', frame)
        cv2.waitKey(3)

  def handle_input_edge_video(self, data, timestamp: float) -> None:
    self.frame_queue.put(data)