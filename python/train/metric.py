import torch.nn.functional as F
from utils.logger import logger
from typing import Any

class CrossEntropyLoss():
  def __init__(self):
    self.loss = 0

  def update(self, output, target):
    print('Loss shape', target.shape)
    self.loss += F.cross_entropy(output, target, reduction="sum").item()

  def compute(self):
    pass

  def reset(self):
    self.loss = 0

class Metric():
  def __init__(self):
    self.metric = {}
    self.add_group('train')
    self.add_group('valid')
    self.add_group('test')

  def add_group(self, group:str):
    if group in self.metric:
      logger.error(f'Group {group} has already exist in the metric class.')
      return
    self.metric[group] = {}

  def add_metric(self, group:str, name:str, metric):
    if name in self.metric[group]:
      logger.error(f'Metric {metric} has already exist in the Group {group}.')
      return
    if group not in self.metric:
      logger.error(f'Group {group} not found in the metric class.')
      return
    self.metric[group][name] = metric

  def update(self, output, target, group:str=None):
    # metric(output, target)  metric.update(output,target)
    for g in self.metric:
      if group is None or g == group:
        for name in self.metric[group]:
          try:
            self.metric[group][name](output, target)
            continue
          except:
            pass
          try:
            self.metric[group][name].update(output, target)
            continue
          except:
            pass

  def compute(self, group:str=None):
    # metric.compute()
    for g in self.metric:
      if group is None or g == group:
        for name in self.metric[group]:
          try:
            self.metric[group][name].compute()
          except:
            pass

  def reset(self, group:str=None):
    # metric.reset()
    for g in self.metric:
      if group is None or g == group:
        for name in self.metric[group]:
          try:
            self.metric[group][name].reset()
          except:
            pass
