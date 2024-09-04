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

    TELLO_PORT = 8889
    COMMAND_PORT = 9000
    VIDEO_PORT = 11111

    def __init__(
      self,
      name: str,
      graph: Graph,
      ip: str,
      ar_video_ip: int = '192.168.3.12',
      ar_video_port: int = 11111,
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
        self.tello_ip = ip 
        self.ar_video_ip = ar_video_ip
        self.ar_video_port = ar_video_port
        self.ar_video_socket = None

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
        self.on_pair()
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_socket.bind(('0.0.0.0', self.COMMAND_PORT))
        Thread(target=self._perform_action).start()
        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.video_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.video_socket.bind(('0.0.0.0', self.VIDEO_PORT))
        Thread(target=self._receive_video).start()
        self.ar_video_socket= socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ar_video_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.action_queue.put('command')
        self.action_queue.put('battery?')
        self.action_queue.put('streamon')
        self.on_connect()

    def _receive_video(self):
        decoder = h264decoder.H264Decoder()
        while True:
            response = self.video_socket.recv(4096)
            if self.ar_video_socket is not None:
                self.ar_video_socket.sendto(response, (self.ar_video_ip, self.ar_video_port))
            framedatas = decoder.decode(response)
            for framedata in framedatas:
                (frame, w, h, ls) = framedata
                array = np.frombuffer(frame, dtype=np.uint8).reshape((h, w, 3))
                cv_frame = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
                self.output(self.OUTPUT_EDGE_VIDEO, cv_frame)

    @override
    def disconnect(self) -> None:
        self.on_disconnect()
        if self.udp_socket is not None:
            self.udp_socket.close()
        if self.video_socket is not None:
            self.video_socket.close()
        if self.ar_video_socket is not None:
            self.ar_video_socket.close()

    @override
    def reconnect(self) -> None:
        self.disconnect()
        self.connect()

    def _perform_action(self):
        while True:
            action = self.action_queue.get()
            if self.udp_socket is not None:
                try:
                    self.udp_socket.sendto(action.encode(), (self.tello_ip, self.TELLO_PORT))
                    response, _ = self.udp_socket.recvfrom(1024)
                    print(action, response.decode())
                    self.output(self.OUTPUT_EDGE_ACTION_RESPONSE, (action, response.decode()))
                except Exception as e:
                    print(e)

    def handle_input_edge_action(self, data: str, timestamp: float) -> None:
        print('Tello', data)
        self.action_queue.put(data)
