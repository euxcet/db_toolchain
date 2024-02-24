import socket
import struct
from typing_extensions import override
from db_graph.framework.graph import Graph
from db_graph.framework.node import Node

class TcpClient(Node):

  OUTPUT_EDGE_DATA = 'data'
  PACKET_HEADER_LEN = 3
  
  def __init__(
      self,
      name: str,
      graph: Graph,
      input_edges: dict[str, str],
      output_edges: dict[str, str],
      address: str,
      port: int,
  ) -> None:
    super(TcpClient, self).__init__(
      name=name,
      graph=graph,
      input_edges=input_edges,
      output_edges=output_edges,
    )
    self.address = address
    self.port = port
    self.buffer = bytearray()
    self.socket = socket.socket()

  @override
  def start(self) -> None:
    self.graph.executor.submit(self.connect())

  def connect(self) -> None:
    self.socket.connect((self.address, self.port))
    while True:
      data = self.socket.recv(1024)
      self.buffer += data
      while True:
        if len(self.buffer) < self.PACKET_HEADER_LEN:
          break
        packet_header = struct.unpack('!3I', self.buffer[:self.PACKET_HEADER_LEN])
        body_size = packet_header[1]
        if len(self.buffer) < self.PACKET_HEADER_LEN + body_size:
          break
        body = self.buffer[self.PACKET_HEADER_LEN : self.PACKET_HEADER_LEN + body_size]
        self.output(self.OUTPUT_EDGE_DATA, body)
        self.buffer = self.buffer[self.PACKET_HEADER_LEN + body_size:]
