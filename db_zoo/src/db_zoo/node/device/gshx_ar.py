import cv2
import queue
import socket
import h264decoder
import numpy as np
from threading import Thread
from typing_extensions import override
from db_graph.framework.graph import Graph
from db_graph.framework.device import Device, DeviceLifeCircleEvent
from db_graph.utils.logger import logger

class GshxAR(Device):
    INPUT_EDGE_VIDEO = 'video'
    OUTPUT_EDGE_EYE = 'eye'
    OUTPUT_EDGE_LIFECYCLE = 'lifecycle'

    def __init__(
      self,
      name: str,
      graph: Graph,
      input_edges: dict[str, str] = {},
      output_edges: dict[str, str] = {},
    ) -> None:
        super().__init__(
            name=name,
            graph=graph,
            input_edges=input_edges,
            output_edges=output_edges,
        )

    # lifecycle callbacks
    @override
    def on_pair(self) -> None:
        self.log_info(f"GSHX AR: Pairing")
        self.lifecycle_status = DeviceLifeCircleEvent.on_pair
        self.output(self.OUTPUT_EDGE_LIFECYCLE, DeviceLifeCircleEvent.on_pair)

    @override
    def on_connect(self) -> None:
        self.log_info("GSHX AR: Connected")
        self.lifecycle_status = DeviceLifeCircleEvent.on_connect
        self.output(self.OUTPUT_EDGE_LIFECYCLE, DeviceLifeCircleEvent.on_connect)

    @override
    def on_disconnect(self, *args, **kwargs) -> None:
        self.log_info("GSHX AR: Disconnected")
        self.lifecycle_status = DeviceLifeCircleEvent.on_disconnect
        self.output(self.OUTPUT_EDGE_LIFECYCLE, DeviceLifeCircleEvent.on_disconnect)

    @override
    def on_error(self) -> None:
        self.lifecycle_status = DeviceLifeCircleEvent.on_error
        self.output(self.OUTPUT_EDGE_LIFECYCLE, DeviceLifeCircleEvent.on_error)

    @override
    def connect(self) -> None:
        ...


    @override
    def disconnect(self) -> None:
        ...

    @override
    def reconnect(self) -> None:
        self.disconnect()
        self.connect()