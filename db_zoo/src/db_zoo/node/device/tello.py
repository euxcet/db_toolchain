import queue
import socket
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
        self.action_queue = queue.Queue[str]()
        self.tello_ip = '192.168.10.1'
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
        self.udp_socket.bind(('0.0.0.0', 9000))
        Thread(target=self._perform_action).start()

    @override
    def disconnect(self) -> None:
        if self.udp_socket is not None:
            self.udp_socket.close()

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
