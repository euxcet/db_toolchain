import time
import socket
import struct
from typing_extensions import override
from typing import Any
from db_graph.framework.graph import Graph
from db_graph.framework.node import Node
from ...utils.byte import to_bytearray

class TcpServer(Node):

  INPUT_EDGE_DATA  = 'data'

  def __init__(
      self,
      name: str,
      graph: Graph,
      input_edges: dict[str, str],
      output_edges: dict[str, str],
      address: str,
      port: int,
      max_client_num: int,
  ) -> None:
    super(TcpServer, self).__init__(
      name=name,
      graph=graph,
      input_edges=input_edges,
      output_edges=output_edges,
    )
    self.address = address
    self.port = port
    self.server = socket.socket()
    self.server.bind((address, port))
    self.server.listen(max_client_num)
    self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    self.server.setblocking(False)
    self.clients:list[socket.socket] = []

  @override
  def start(self) -> None:
    self.graph.executor.submit(self.connect())

  def connect(self) -> None:
    while True:
      try:
        sock, address = self.server.accept()
        self.clients.append(sock)
        self.log_info(f'Connected with {address[0]}:{address[1]}')
      except BlockingIOError:
        time.sleep(0.1)

  def clean_clients(self):
    for client in self.clients:
      if client.fileno() == -1:
        self.clients.remove(client)

  def handle_input_edge_data(self, data: Any) -> None:
    packet = bytearray()
    packet_body = to_bytearray(data)
    packet_header = struct.pack('!3I', [0x0, len(packet_body), 0x0])
    packet += packet_header + packet_body
    for client in self.clients:
      client.send(packet)