import argparse
from abc import ABCMeta, abstractmethod
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.classification import Accuracy
from torch.utils.data import DataLoader, random_split
from lightning.fabric import Fabric, seed_everything
from train.metric import MetricGroup

class Trainer(metaclass=ABCMeta):
  def __init__(self, num_classes:int, batch_size:int, epochs:int, lr:float, gamma:float, seed:int, log_interval:int,
               train_ratio:float, valid_ratio:float, dataset_path:float, output_model_name:float, **kwargs):
    self.num_classes = num_classes
    self.batch_size = batch_size
    self.epochs = epochs
    self.lr = lr
    self.gamma = gamma
    self.seed = seed
    self.log_interval = log_interval
    self.train_ratio = train_ratio
    self.valid_ratio = valid_ratio
    self.dataset_path = dataset_path.strip().split(',')
    self.output_model_name = output_model_name

    self.best_metric = None
    self.best_metric_updated = False

    seed_everything(seed)
    self.fabric = Fabric(accelerator="auto")
    self.init_dataset()
    self.init_model()
    self.init_metric()

  def train(self):
    for epoch in range(1, self.epochs + 1):
      self.train_epoch(epoch)
      if epoch % self.log_interval == 0:
        self.log_epoch(epoch)

  def update_best_metric(self, metric) -> bool:
    if self.best_metric is None or metric > self.best_metric:
      self.best_metric = deepcopy(metric)
      self.best_metric_updated = False
      return True
    self.best_metric_updated = True
    return False

  def save_model(self, model=None):
    if model is None:
      try:
        torch.save(self.model.state_dict(), self.output_model_name)
      except:
        pass
    else:
      torch.save(model.state_dict(), self.output_model_name)

  def _default_backward(self):
    # The subclass should include the variables model, metric, optimizer, criterion, scheduler and train_loader.
    self.model.train()
    self.metric.reset(group='Train')
    for batch_id, (data, target) in enumerate(self.train_loader):
      self.optimizer.zero_grad()
      output = self.model(data)
      loss = self.criterion(output, target)
      self.metric.update(output, target, group='Train')
      self.fabric.backward(loss)
      self.optimizer.step()
    self.metric.compute(group='Train')
    self.scheduler.step()

  def _default_valid(self):
    # The subclass should include the variables model, metric and valid_loader.
    self.model.eval()
    self.metric.reset(group='Valid')
    with torch.no_grad():
      for data, target in self.valid_loader:
        output = self.model(data)
        self.metric.update(output, target, group='Valid')
    self.metric.compute(group='Valid')

  def _default_test(self):
    # The subclass should include the variables model, metric and test_loader.
    self.model.eval()
    self.metric.reset(group='Test')
    with torch.no_grad():
      for data, target in self.test_loader:
        output = self.model(data)
        self.metric.update(output, target, group='Test')
    self.metric.compute(group='Test')

  @abstractmethod
  def init_dataset(self): ...

  @abstractmethod
  def init_model(self): ...

  @abstractmethod
  def init_metric(self): ...
    
  @abstractmethod
  def train_epoch(self, epoch:int): ...

  @abstractmethod
  def log_epoch(self, epoch:int): ...

def add_argument(parser:argparse.ArgumentParser):
  parser.add_argument("--num-classes", type=int, metavar="N", help="number of classes")
  parser.add_argument("--batch-size", type=int, default=32, metavar="N", help="input batch size for training (default: 32)")
  parser.add_argument("--epochs", type=int, default=50, metavar="N", help="number of epochs to train (default: 50)")
  parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 1.0)")
  parser.add_argument("--gamma", type=float, default=0.1, metavar="M", help="learning rate step gamma (default: 0.7)")
  parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
  parser.add_argument("--train-ratio", type=float, default=0.8, metavar="R", help="ratio of training dataset size")
  parser.add_argument("--dataset-path", type=str, metavar="PATH", default='./local_dataset/dataset', help="dataset location")
  parser.add_argument("--valid-ratio", type=float, default=0.2, metavar="R", help="ratio of valid dataset size")
  parser.add_argument("--log-interval", type=int, default=1, metavar="I", help="Print log every few times")
  parser.add_argument("--output-model-name", type=str, metavar="PATH", default="best.pth", help="Name of the output model")