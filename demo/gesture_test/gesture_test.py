import os
import time
import pygame
import random
from typing_extensions import override
from db_graph.framework.node import Node
from db_graph.framework.graph import Graph

class GestureTest(Node):

  INPUT_EDGE_GESTURE = 'imu'
  MIN_TIME = 500
  MAX_TIME = 2000

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
    self.waiting = False

  @override
  def start(self) -> None:
    os.makedirs('result', exist_ok=True)
    self.fout = open('result/' + str(int(time.time())) + '.txt', 'w')

  @override
  def block(self) -> None:
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    clock = pygame.time.Clock()
    running = True
    font = pygame.font.Font(size=70)
    text = font.render("Test", True, (255, 255, 255), (0, 0, 0))
    
    self.task_time = time.time()
    self.gap_time = random.randint(self.MIN_TIME, self.MAX_TIME)

    while running:
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and pygame.key.name(event.key) == 'space':
          self.trigger('keyboard')
      if not self.waiting and time.time() - self.task_time > self.gap_time / 1000:
        self.begin_time = time.time()
        self.waiting = True
      screen.fill("black")
      if self.waiting:
        screen.blit(text, (280, 200))
      pygame.display.flip()
      clock.tick(60)
    pygame.quit()

  def trigger(self, source: str):
    if self.waiting:
      print(time.time() - self.begin_time, source)
      self.fout.write(str(time.time() - self.begin_time) + ' ' + source + '\n')
      self.fout.flush()
      self.task_time = time.time()
      self.gap_time = random.randint(self.MIN_TIME, self.MAX_TIME)
      self.waiting = False

  def handle_input_edge_gesture(self, data: str, timestamp: float) -> None:
    self.trigger('ring')
