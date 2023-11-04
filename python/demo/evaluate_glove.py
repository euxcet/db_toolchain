import os
import time
import queue
import torch
import socket
import struct
import asyncio
import argparse
import numpy as np
from pynput import keyboard
from threading import Timer, Thread
from python.model.fully_connected import FullyConnectedModel

class Pose:
  def __init__(self, qw, qx, qy, qz):
    self.qw = qw
    self.qx = qx
    self.qy = qy
    self.qz = qz

class Glove(Thread):
  def __init__(self, ip:str, out_ip:str=""):
    Thread.__init__(self)
    self.ip = ip
    self.out_ip = out_ip
    self.first_timestamp = 0
    self.count = 0
    self.save_file_name = None
    self.streaming = True
    self.model = FullyConnectedModel(11)
    self.model.load_state_dict(torch.load('glove.pth'))
    self.model.eval()

  def parse_pose_from_data(self, data):
    qw, = struct.unpack('f', data[0:4])
    qx, = struct.unpack('f', data[4:8])
    qy, = struct.unpack('f', data[8:12])
    qz, = struct.unpack('f', data[12:16])
    return Pose(qw, qx, qy, qz)

  def process_data(self, data):
    if data.decode('cp437').find('VRTRIX') == 0:
      radioStrength, = struct.unpack('h', data[265:267])
      battery, = struct.unpack('f', data[267:271])
      calScore, = struct.unpack('h', data[271:273])
      joint_quat = []
      for i in range(0, 16):
        joint_quat.append(self.parse_pose_from_data(data[9 + 16 * i : 25 + 16 * i])) 

      current_time = time.time()
      if self.first_timestamp == 0:
        self.first_timestamp = current_time
      else:
        self.count += 1
        if current_time > self.first_timestamp + 10:
          print('RadioStrength: {}, Battery: {}, CalScore: {}, FPS: {:.2f}'.format(
            -radioStrength, battery, calScore, self.count / (current_time - self.first_timestamp)))
          self.first_timestamp = 0
          self.count = 0

      if self.count % 10 == 0:
        input_data = torch.tensor(np.array([[j.qw, j.qx, j.qy, j.qz] for j in joint_quat]).astype(np.float32).reshape(1, 64))
        output = self.model(input_data).cpu().detach().numpy().flatten()
        gesture_id = np.argmax(output)
        print(gesture_id, output[gesture_id])


  def run(self):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (self.ip, 11002)
    print('Connecting to %s port %s' % server_address)
    sock.connect(server_address)
    print('Glove connected.')

    out_sock = None
    client_sock = None

    if self.out_ip != "":
      out_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      out_server_address = (self.out_ip, 11002)
      print('output ip %s port %s' % out_server_address)
      out_sock.bind(out_server_address)
      out_sock.listen(2)
      client_sock, client_address = out_sock.accept()
      print('Glove out connected.')

    while self.streaming:
      data = sock.recv(1024)
      if client_sock is not None:
        client_sock.send(data)
      self.process_data(data)

if __name__ == '__main__':
  glove = Glove('192.168.1.16')
  glove.run()
