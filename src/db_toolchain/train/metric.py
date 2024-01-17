from __future__ import annotations
import torch.nn.functional as F
from utils.logger import logger
from typing import Any
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
from torchmetrics.classification import Accuracy, ConfusionMatrix
from enum import Enum

class Metric(metaclass=ABCMeta):
  def __init__(self):
    self.value = 0

  @abstractmethod
  def update(self, output, target): ...

  @abstractmethod
  def compute(self): ...

  @abstractmethod
  def reset(self): ...

class CrossEntropyLoss(Metric):
  def __init__(self):
    super(CrossEntropyLoss, self).__init__()
    self.sum = 0
    self.len = 0

  def update(self, output, target):
    self.sum += F.cross_entropy(output, target, reduction="sum").item()
    self.len += target.shape[0]

  def compute(self):
    self.value = 0 if self.len == 0 else self.sum / self.len

  def reset(self):
    self.value = 0
    self.sum = 0
    self.len = 0

  def __lt__(self, other:CrossEntropyLoss):
    return self.value < other.value

  def __str__(self):
    return f'{self.value:.4f}'

class TrajectoryLoss(Metric):
  def __init__(self):
    super(TrajectoryLoss, self).__init__()
    self.sum = 0
    self.len = 0
    self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-8)

  def update(self, output, target):
    length_loss = torch.mean(torch.abs((torch.norm(output, dim=1) - torch.norm(target, dim=1))))
    angle_loss = 1 - torch.mean(self.cos_sim(output, target))
    loss = length_loss + angle_loss * 2
    self.sum += loss.item() * target.shape[0]
    self.len += target.shape[0]

  def compute(self):
    self.value = 0 if self.len == 0 else self.sum / self.len

  def reset(self):
    self.value = 0
    self.sum = 0
    self.len = 0

  def __lt__(self, other:CrossEntropyLoss):
    return self.value < other.value

  def __str__(self):
    return f'{self.value:.4f}'

class AccuracyMetric(Metric):
  def __init__(self, device, **kwargs):
    super(AccuracyMetric, self).__init__()
    self.metric = Accuracy(**kwargs).to(device)

  def update(self, output, target):
    self.metric.update(output, target)

  def compute(self):
    self.value = self.metric.compute().float()

  def reset(self):
    self.metric.reset()
    self.value = 0
  
  def __lt__(self, other:AccuracyMetric):
    return self.value < other.value

  def __str__(self):
    return f'{100 * self.value:.2f}%'

class ConfusionMatrixMetric(Metric):
  def __init__(self, device, **kwargs):
    super(ConfusionMatrixMetric, self).__init__()
    self.metric = ConfusionMatrix(**kwargs).to(device)

  def update(self, output, target):
    self.metric.update(output, target)

  def compute(self):
    self.value = self.metric.compute()

  def reset(self):
    self.metric.reset()

  def plot(self):
    fig, ax = self.metric.plot()
    # fig.savefig('output.png')
  
  def __lt__(self, other:ConfusionMatrixMetric):
    raise NotImplementedError()

  def __str__(self):
    return "<Not Displayed>"

class MetricGroup():
  def __init__(self, lt_func):
    self.metric:dict[str, dict[str, Metric]] = {}
    self.lt_func = lt_func
    self.add_group('Train')
    self.add_group('Valid')
    self.add_group('Test')

  def __getitem__(self, key):
    return self.metric[key]

  def __lt__(self, other):
    return self.lt_func(self, other)

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

  def iter(self, group:str=None, name:str=None):
    for g in self.metric:
      if group is None or g == group:
        for n in self.metric[group]:
          if name is None or n == name:
            yield self.metric[g][n]

  def update(self, output, target, group:str=None):
    for metric in self.iter(group=group):
      metric.update(output, target)

  def compute(self, group:str=None):
    for metric in self.iter(group=group):
      metric.compute()

  def reset(self, group:str=None):
    for metric in self.iter(group=group):
      metric.reset()

  def __str__(self):
    return '\n'.join([
      g + '\t' + '  '.join(n + ' ' + str(self.metric[g][n]) for n in self.metric[g])
        for g in filter(lambda x: len(self.metric[x]) > 0, self.metric)
    ])
