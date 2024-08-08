import cv2
import time
import queue
import socket
import h264decoder
import numpy as np
from threading import Thread
from typing_extensions import override
from db_graph.framework.graph import Graph
from db_graph.framework.device import Device, DeviceLifeCircleEvent
from db_graph.utils.logger import logger
from ...utils.packet_wrapper import wrap, PacketType

class GshxAR(Device):
    INPUT_EDGE_OBJECTS = 'objects'
    OUTPUT_EDGE_LIFECYCLE = 'lifecycle'
    OUTPUT_EDGE_EYE = 'eye'

    AR_PORT = 9001

    def __init__(
      self,
      name: str,
      graph: Graph,
      input_edges: dict[str, str] = {},
      output_edges: dict[str, str] = {},
      ip: str = '192.168.3.12'
    ) -> None:
        super().__init__(
            name=name,
            graph=graph,
            input_edges=input_edges,
            output_edges=output_edges,
        )
        self.ar_socket = None
        self.ar_ip = ip
        print(self.ar_ip)

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
        self.ar_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ar_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.ar_socket.bind(('0.0.0.0', self.AR_PORT))
        Thread(target=self._keep_alive).start()
        Thread(target=self._get_data).start()

    def _send(self, type: PacketType, data: bytes = bytes()) -> None: 
        if self.ar_socket is not None and self.ar_ip is not None:
            self.ar_socket.sendto(wrap(type, data), (self.ar_ip, self.AR_PORT))

    def _keep_alive(self) -> None:
        while True:
            self._send(PacketType.HEARTBEAT)
            time.sleep(1)

    def _get_data(self) -> None:
        while True:
            response, addr = self.ar_socket.recvfrom(4096)
            print(response, addr)

    @override
    def disconnect(self) -> None:
        ...

    def handle_input_edge_objects(self, data: bytes, timestamp: float) -> None:
        self._send(PacketType.OBJECTS, data)

    @override
    def reconnect(self) -> None:
        self.disconnect()
        self.connect()