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

class Tello(Device):
    INPUT_EDGE_ACTION = 'action'
    OUTPUT_EDGE_ACTION_RESPONSE = 'action_response'
    OUTPUT_EDGE_BATTERY = 'battery'
    OUTPUT_EDGE_VIDEO = 'video'
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
        self.udp_socket = None
        self.video_socket = None
        self.action_queue = queue.Queue[str]()
        self.tello_ip = '192.168.3.15'
        self.tello_port = 8889

    # lifecycle callbacks
    @override
    def on_pair(self) -> None:
        self.log_info(f"Tello: Pairing")
        self.lifecycle_status = DeviceLifeCircleEvent.on_pair
        self.output(self.OUTPUT_EDGE_LIFECYCLE, DeviceLifeCircleEvent.on_pair)

    @override
    def on_connect(self) -> None:
        self.log_info("Tello: Connected")
        self.lifecycle_status = DeviceLifeCircleEvent.on_connect
        self.output(self.OUTPUT_EDGE_LIFECYCLE, DeviceLifeCircleEvent.on_connect)

    @override
    def on_disconnect(self, *args, **kwargs) -> None:
        self.log_info("Tello: Disconnected")
        self.lifecycle_status = DeviceLifeCircleEvent.on_disconnect
        self.output(self.OUTPUT_EDGE_LIFECYCLE, DeviceLifeCircleEvent.on_disconnect)

    @override
    def on_error(self) -> None:
        self.lifecycle_status = DeviceLifeCircleEvent.on_error
        self.output(self.OUTPUT_EDGE_LIFECYCLE, DeviceLifeCircleEvent.on_error)

    @override
    def connect(self) -> None:
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_socket.bind(('0.0.0.0', 9000))
        Thread(target=self._perform_action).start()
        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.video_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.video_socket.bind(('0.0.0.0', 11111))
        Thread(target=self._receive_video).start()
        self.broadcast_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.broadcast_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.broadcast_socket.bind(('0.0.0.0', 11112))

    def _receive_video(self):
        decoder = h264decoder.H264Decoder()
        while True:
            response = self.video_socket.recv(4096)
            self.broadcast_socket.send(response)
            framedatas = decoder.decode(response)
            for framedata in framedatas:
                (frame, w, h, ls) = framedata
                array = np.frombuffer(frame, dtype=np.uint8).reshape((720, 960, 3))
                self.output(self.OUTPUT_EDGE_VIDEO, cv2.cvtColor(array, cv2.COLOR_BGR2RGB))

    @override
    def disconnect(self) -> None:
        self.on_disconnect()
        if self.udp_socket is not None:
            self.udp_socket.close()
        if self.video_socket is not None:
            self.video_socket.close()
        if self.broadcast_socket is not None:
            self.broadcast_socket.close()
        # connect()

    @override
    def reconnect(self) -> None:
        self.disconnect()
        self.connect()

    def _perform_action(self):
        while True:
            action = self.action_queue.get()
            if self.udp_socket is not None:
                self.udp_socket.sendto(action.encode(), (self.tello_ip, self.tello_port))
                response, _ = self.udp_socket.recvfrom(1024)
                self.output(self.OUTPUT_EDGE_ACTION_RESPONSE, (action, response.decode()))

    def handle_input_edge_action(self, data: str, timestamp: float) -> None:
        self.action_queue.put(data)
