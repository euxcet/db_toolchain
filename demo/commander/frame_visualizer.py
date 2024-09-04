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

class FrameVisualizer(Node):

  INPUT_EDGE_VIDEO = 'video'
  INPUT_EDGE_OBJECTS = 'objects'
  INPUT_EDGE_FACES = 'faces'

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
    self.objects = None
    self.faces = None

  @override
  def start(self):
    ...

  @override
  def block(self):
    while True:
        frame = self.frame_queue.get()
        if self.objects is not None:
            for box, conf in zip(self.objects.xyxy, self.objects.conf):
                if conf > 0.5:
                    frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
        if self.faces is not None:
          locations, id = self.faces
          for box in locations:
            print(box)
            frame = cv2.rectangle(frame, (int(box[1]), int(box[0])), (int(box[3]), int(box[2])), (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        cv2.waitKey(5)

  def handle_input_edge_objects(self, data, timestamp: float) -> None:
    self.objects = data

  def handle_input_edge_faces(self, data, timestamp: float) -> None:
    self.faces = data

  def handle_input_edge_video(self, data, timestamp: float) -> None:
    self.frame_queue.put(data)