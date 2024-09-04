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

class VideoMock(Node):

  OUTPUT_EDGE_VIDEO = 'video'

  def __init__(
      self,
      name: str,
      graph: Graph,
      input_edges: dict[str, str],
      output_edges: dict[str, str],
      path: str,
  ) -> None:
    super().__init__(
      name=name,
      graph=graph,
      input_edges=input_edges,
      output_edges=output_edges,
    )
    self.path = path

  @override
  def start(self):
    Thread(target=self.read_video).start()

  def read_video(self):
    while True:
      capture = cv2.VideoCapture(self.path)
      while True:
        ret, frame = capture.read()
        if not ret:
          break
        self.output(self.OUTPUT_EDGE_VIDEO, frame)
        time.sleep(0.1)

  @override
  def block(self):
    ...