from __future__ import annotations
from typing import Any
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, ConfusionMatrix
from typing_extensions import override
from .utils.logger import logger

class Metric(metaclass=ABCMeta):
  def __init__(self):
    self.value = 0

  @abstractmethod
  def update(self, output: Any, target: Any): ...

  @abstractmethod
  def compute(self): ...

  @abstractmethod
  def reset(self): ...

class CrossEntropyLoss(Metric):
  def __init__(self):
    super(CrossEntropyLoss, self).__init__()
    self.sum = 0
    self.len = 0

  @override
  def update(self, result: tuple):
    output, target = result
    self.sum += F.cross_entropy(output, target, reduction="sum").item()
    self.len += target.shape[0]

  @override
  def compute(self):
    self.value = 0 if self.len == 0 else self.sum / self.len

  @override
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

  @override
  def update(self, result: tuple):
    output, target = result
    length_loss = torch.mean(torch.abs((torch.norm(output, dim=1) - torch.norm(target, dim=1))))
    angle_loss = 1 - torch.mean(self.cos_sim(output, target))
    loss = length_loss + angle_loss * 2
    self.sum += loss.item() * target.shape[0]
    self.len += target.shape[0]

  @override
  def compute(self):
    self.value = 0 if self.len == 0 else self.sum / self.len

  @override
  def reset(self):
    self.value = 0
    self.sum = 0
    self.len = 0

  def __lt__(self, other: CrossEntropyLoss):
    return self.value < other.value

  def __str__(self):
    return f'{self.value:.4f}'

class AccuracyMetric(Metric):
  def __init__(self, device, **kwargs):
    super(AccuracyMetric, self).__init__()
    self.metric = Accuracy(**kwargs).to(device)

  @override
  def update(self, result: tuple):
    output, target = result
    self.metric.update(output, target)

  @override
  def compute(self):
    self.value = self.metric.compute().float()

  @override
  def reset(self):
    self.metric.reset()
    self.value = 0
  
  def __lt__(self, other: AccuracyMetric):
    return self.value < other.value

  def __str__(self):
    return f'{100 * self.value:.2f}%'

class ConfusionMatrixMetric(Metric):
  def __init__(self, device, **kwargs):
    super(ConfusionMatrixMetric, self).__init__()
    self.metric = ConfusionMatrix(**kwargs).to(device)

  @override
  def update(self, result: tuple):
    output, target = result
    self.metric.update(output, target)

  @override
  def compute(self):
    self.value = self.metric.compute()

  @override
  def reset(self):
    self.metric.reset()

  def plot(self):
    fig, ax = self.metric.plot()
    # fig.savefig('output.png')
  
  def __lt__(self, other: ConfusionMatrixMetric):
    raise NotImplementedError()

  def __str__(self):
    return "<Not Displayed>"

class MetricGroup():
  def __init__(self, lt_func: function = lambda x, other: x['Valid']['Accuracy'] < other['Valid']['Accuracy'], wandb_logger=None) -> None:
    self.metric:dict[str, dict[str, Metric]] = {}
    self.lt_func = lt_func
    self.wandb_logger = wandb_logger
    self.add_groups(['Train', 'Valid', 'Test'])

  def set_lt_func(self, lt_func: function) -> None:
    self.lt_func = lt_func

  def __getitem__(self, key):
    return self.metric[key]

  def __lt__(self, other: MetricGroup):
    return self.lt_func(self, other)

  def add_group(self, group: str):
    if group in self.metric:
      logger.error(f'Group {group} has already exist in the metric class.')
      return
    self.metric[group] = {}

  def add_groups(self, groups: list[str]):
    for group in groups:
      self.add_group(group)

  def add_metric(self, group: str, name: str, metric: Metric):
    if name in self.metric[group]:
      logger.error(f'Metric {name} has already exist in the Group {group}.')
      return
    if group not in self.metric:
      logger.error(f'Group {group} not found in the metric class.')
      return
    self.metric[group][name] = metric

  def iter(self, group: str = None, name: str = None):
    for g in self.metric:
      if group is None or g == group:
        for n in self.metric[group]:
          if name is None or n == name:
            yield self.metric[g][n]

  def update(self, result: tuple , group: str = None):
    for metric in self.iter(group=group):
      metric.update(result)

  def compute(self, group: str = None):
    for metric in self.iter(group=group):
      metric.compute()

  def reset(self, group: str = None):
    for metric in self.iter(group=group):
      metric.reset()

  def log_to_wandb(self, step: int, group: str = 'Valid'):
    if self.wandb_logger is not None:
      for metric in self.iter(group=group):
        self.wandb_logger.log({f'{group}/{metric}': metric.value}, step=step)

  def __str__(self):
    return '\n'.join([
      g + '\t' + '  '.join(n + ' ' + str(self.metric[g][n]) for n in self.metric[g])
        for g in filter(lambda x: len(self.metric[x]) > 0, self.metric)
    ])

  def to_dict(self):
    return {f"{g}/{n}": self.metric[g][n].value for g in filter(lambda x: len(self.metric[x]) > 0, self.metric) for n in self.metric[g]}