import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from model.imu_trajectory_model import TrajectoryLSTMModel
from train.imu_trajectory_dataset import get_trajectory_dataset

from train.trainer import Trainer, add_argument
from train.metric import MetricGroup, TrajectoryLoss

class Loss(nn.Module):
  def __init__(self):
    super().__init__()
    self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-8)

  def forward(self, x, y):
    length_loss = torch.mean(torch.abs((torch.norm(x, dim=1) - torch.norm(y, dim=1))))
    angle_loss = 1 - torch.mean(self.cos_sim(x, y))
    return length_loss + angle_loss * 2

class TrajectoryTrainer(Trainer):
  def __init__(self, **kwargs):
    super(TrajectoryTrainer, self).__init__(**kwargs)

  def init_dataset(self):
    self.dataset = get_trajectory_dataset(self.dataset_path)
    self.train_dataset, self.valid_dataset = random_split(self.dataset, [self.train_ratio, 1 - self.train_ratio])
    self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)
    self.train_loader, self.valid_loader = self.fabric.setup_dataloaders(self.train_loader, self.valid_loader)

  def init_model(self):
    self.model = TrajectoryLSTMModel()
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=10)
    self.criterion = Loss()

  def init_metric(self):
    self.metric = MetricGroup(lt_func=lambda x, other: x['Valid']['Loss'] < other['Valid']['Loss'])
    self.metric.add_metric('Train', 'Loss', TrajectoryLoss())
    self.metric.add_metric('Valid', 'Loss', TrajectoryLoss())

  def train_epoch(self, epoch:int):
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
    self.scheduler.step(self.metric['Train']['Loss'].value)
    self._default_valid()
    if self.update_best_metric(self.metric):
      self.save_model()

  def log_epoch(self, epoch:int):
    print(f'Epoch: {epoch}\n{self.metric}\n')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Training code for trajectory models")
  add_argument(parser)
  args = parser.parse_args()

  TrajectoryTrainer(**vars(args)).train()
  # python train_imu_trajectory.py --dataset-path=./local_dataset/trajectory/ringboard_dataset --batch-size=32 --lr=0.003