import time
import struct
import numpy as np
from typing_extensions import override
from db_graph.data.imu_data import IMUData
from db_graph.framework.graph import Graph
from db_graph.framework.node import Node
from ultralytics.engine.results import Boxes

class Commander(Node):

  INPUT_EDGE_EYE = 'eye'
  INPUT_EDGE_OBJECTS = 'objects'
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

  @override
  def start(self):
    ...

  def handle_input_edge_eye(self, data, timestamp: float) -> None:
    print(data)

  def handle_input_edge_objects(self, boxes: Boxes, timestamp: float) -> None:
    if boxes.id is not None:
        data = struct.pack('h', boxes.id.shape[0])
        for id, cls, conf, xyxyn in zip(boxes.id, boxes.cls, boxes.conf, boxes.xyxyn):
            id, cls, conf, xyxyn = id.numpy(), cls.numpy(), conf.numpy(), xyxyn.numpy()
            data += struct.pack('IIfffff', int(id), int(cls), conf, xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3])
        self.output(self.OUTPUT_EDGE_OBJECTS, data)
    else:
        data = struct.pack('h', 0)
        self.output(self.OUTPUT_EDGE_OBJECTS, data)
