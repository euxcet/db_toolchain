import struct
import numpy as np
from typing import Any
from typing_extensions import override
from db_graph.framework.graph import Graph
from db_graph.framework.node import Node
from db_graph.data.imu_data import IMUData
from collections import deque
import vispy.app
from vispy import scene, color
import pyquaternion as pyq
from ...utils.plotter import MultiPlotter

class OrientationVisualizer(Node):

  INPUT_EDGE_ORIENTATION = 'orientation'
  
  def __init__(
      self,
      name: str,
      graph: Graph,
      input_edges: dict[str, str],
      output_edges: dict[str, str],
  ) -> None:
    super(OrientationVisualizer, self).__init__(
      name=name,
      graph=graph,
      input_edges=input_edges,
      output_edges=output_edges,
    )
    canvas = scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    view.camera.aspect = 1
    self.pos = np.array([[0, 0, 0], [0, 1, 0]], dtype='float32')
    self.lines = scene.Line(pos=self.pos, connect=np.array([[0,1]]),
            antialias=False, method='gl', color=(1, 1, 1, 1), parent=view.scene)
    self.markers = scene.Markers(pos=self.pos,face_color=np.array([[1,1,1],[1,1,1]]), 
                symbol='o',parent=view.scene) 
    view.camera.set_range() 

    def update(ev):
      self.lines.set_data(self.pos)
      self.markers.set_data(self.pos)

    timer = vispy.app.Timer(interval=0.01, connect=update, start=True)

    grid = scene.visuals.GridLines(parent=view.scene)

    # 创建三维坐标轴
    axis = scene.visuals.XYZAxis(parent=view.scene)

    # 设置视图相机的方向和范围
    view.camera = scene.cameras.TurntableCamera(elevation=10, azimuth=10, distance=5)

  @override
  def start(self) -> None:
    ...

  @override
  def block(self) -> None:
    vispy.app.run()

  def handle_input_edge_orientation(self, data: pyq.Quaternion, timestamp: float) -> None:
    # print('ori', data)
    self.pos = np.array([[0, 0, 0], data.rotate([1, 0, 0])], dtype='float32')