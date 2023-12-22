import argparse
from abc import ABCMeta, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.classification import Accuracy
from torch.utils.data import DataLoader, random_split
from lightning.fabric import Fabric, seed_everything

class Trainer(metaclass=ABCMeta):
  def __init__(self, args):
    self.args = args
    self.fabric = Fabric(accelerator="auto")
    seed_everything(args.seed)
    self.init_dataset()
    self.init_model()
    self.init_metric()
    self.train()

  def train(self):
    for epoch in range(1, self.args.epochs + 1):
      self.train_epoch(epoch)

  @abstractmethod
  def init_dataset(self):
    pass

  @abstractmethod
  def init_model(self):
    pass

  @abstractmethod
  def init_metric(self):
    pass
    
  @abstractmethod
  def train_epoch(self, epoch:int):
    pass


def add_argument(parser:argparse.ArgumentParser):
  parser.add_argument("--batch-size", type=int, default=32, metavar="N", help="input batch size for training (default: 32)")
  parser.add_argument("--epochs", type=int, default=50, metavar="N", help="number of epochs to train (default: 50)")
  parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 1.0)")
  parser.add_argument("--gamma", type=float, default=0.1, metavar="M", help="learning rate step gamma (default: 0.7)")
  parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
  parser.add_argument("--train-ratio", type=float, default=0.8, metavar="R", help="ratio of training dataset size")
  parser.add_argument("--dataset-path", type=str, metavar="PATH", default='./local_dataset/trajectory/touch_dataset', help="dataset location")
  parser.add_argument("--num-classes", type=int, metavar="N", required=True, help="number of classes")
  parser.add_argument("--save-model-path", type=str, metavar="PATH", default="imu_gesture.pth", help="save path of the output model")