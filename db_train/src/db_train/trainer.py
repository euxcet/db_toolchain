from __future__ import annotations

import argparse
from typing import Any
from copy import deepcopy
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader
from lightning.fabric import Fabric, seed_everything

from .metric import MetricGroup

class TrainerParameter():
  def __init__(
      self,
      dataset_path: str,
      output_model_name: float,
      seed: int,
      num_classes: int,
      batch_size: int,
      epochs: int,
      lr: float,
      gamma: float,
      log_interval: int,
      train_ratio: float,
      valid_ratio: float,
  ) -> None:
    self.num_classes = num_classes
    self.batch_size = batch_size
    self.epochs = epochs
    self.lr = lr
    self.gamma = gamma
    self.seed = seed
    self.log_interval = log_interval
    self.train_ratio = train_ratio
    self.valid_ratio = valid_ratio
    self.test_ratio = 1.0 - train_ratio - valid_ratio
    self.dataset_path = dataset_path
    self.output_model_name = output_model_name

  @staticmethod
  def from_dict(param: dict) -> TrainerParameter:
    return TrainerParameter(**param)

class Trainer(ABC):
  def __init__(
      self,
      parameter: TrainerParameter,
  ) -> None:
    self.parameter = parameter


    seed_everything(self.parameter.seed)
    self.fabric = Fabric(accelerator="auto")

    # dataset
    self.dataset: Dataset = None
    self.train_dataset: Dataset = None
    self.valid_dataset: Dataset = None
    self.train_loader: DataLoader = None
    self.valid_loader: DataLoader = None

    # metric
    self.metric: MetricGroup = None
    self.best_metric: MetricGroup = None

    # optimizer
    self.model: nn.Module = None
    self.optimizer: Optimizer = None
    self.scheduler: LRScheduler = None
    self.criterion: nn.Module = None

    self.prepare_dataset()
    self._check_dataset()
    self.prepare_metric()
    self._check_metric()
    self.prepare_optimizers()
    self._check_optimizers()

  def _check_dataset(self):
    ...

  def _check_metric(self):
    ...

  def _check_optimizers(self):
    ...

  def train(self):
    for epoch in range(1, self.parameter.epochs + 1):
      self.training_epoch(epoch)
      if epoch % self.parameter.log_interval == 0:
        self.log_epoch(epoch)

  def backward(self, training_step_func: function = None):
    self.model.train()
    self.metric.reset(group='Train')
    for batch in self.train_loader:
      if training_step_func is None:
        loss, result = self.training_step(batch)
      else:
        loss, result = training_step_func(batch)
      self.optimizer.zero_grad()
      self.metric.update(result, group='Train')
      self.fabric.backward(loss)
      self.optimizer.step()
    self.metric.compute(group='Train')
    self.scheduler.step()

  def validate(self):
    self.model.eval()
    self.metric.reset(group='Valid')
    with torch.no_grad():
      for data, target in self.valid_loader:
        output = self.model(data)
        self.metric.update((output, target), group='Valid')
    self.metric.compute(group='Valid')

  def update_metric(self, metric: MetricGroup = None) -> bool:
    metric = self.metric if metric is None else metric
    if self.best_metric is None or metric > self.best_metric:
      self.best_metric = deepcopy(metric)
      return True
    return False

  def save_model(self, model=None):
    if model is None:
      model = self.model
    torch.save(model.state_dict(), self.parameter.output_model_name)
  
  @abstractmethod
  def prepare_dataset(self): ...

  @abstractmethod
  def prepare_metric(self): ...
    
  @abstractmethod
  def prepare_optimizers(self): ...

  @abstractmethod
  def training_epoch(self, epoch: int): ...

  @abstractmethod
  def training_step(self, batch: Any): ...

  @abstractmethod
  def log_epoch(self, epoch: int): ...

def add_argument(parser: argparse.ArgumentParser):
  parser.add_argument("--dataset-path", required=True, type=str, metavar="PATH", default='./local_dataset/dataset', help="dataset location")
  parser.add_argument("--num-classes", type=int, metavar="N", help="number of classes")
  parser.add_argument("--batch-size", type=int, default=32, metavar="N", help="input batch size for training (default: 32)")
  parser.add_argument("--epochs", type=int, default=50, metavar="N", help="number of epochs to train (default: 50)")
  parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 1.0)")
  parser.add_argument("--gamma", type=float, default=0.1, metavar="M", help="learning rate step gamma (default: 0.7)")
  parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
  parser.add_argument("--train-ratio", type=float, default=0.8, metavar="R", help="ratio of training dataset size")
  parser.add_argument("--valid-ratio", type=float, default=0.2, metavar="R", help="ratio of valid dataset size")
  parser.add_argument("--log-interval", type=int, default=1, metavar="I", help="Print log every few times")
  parser.add_argument("--output-model-name", type=str, metavar="PATH", default="best.pth", help="Name of the output model")