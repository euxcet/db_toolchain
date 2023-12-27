import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))

import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from model.imu_gesture_model import GestureNetCNN
from train.imu_gesture_dataset import get_imu_gesture_dataset

from train.trainer import Trainer, add_argument
from train.metric import MetricGroup, CrossEntropyLoss, AccuracyMetric, ConfusionMatrixMetric

class IMUGestureTrainer(Trainer):
  def __init__(self, **kwargs):
    super(IMUGestureTrainer, self).__init__(**kwargs)

  def init_dataset(self):
    self.dataset = get_imu_gesture_dataset(self.dataset_path)
    self.train_dataset, self.valid_dataset = random_split(self.dataset, [self.train_ratio, 1 - self.train_ratio])
    self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)
    self.train_loader, self.valid_loader = self.fabric.setup_dataloaders(self.train_loader, self.valid_loader)

  def init_model(self):
    self.model = GestureNetCNN(num_classes=self.num_classes)
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
    self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5, 8], gamma=self.gamma)
    self.criterion = nn.CrossEntropyLoss()

  def init_metric(self):
    self.metric = MetricGroup(lt_func=lambda x, other: x['Valid']['Accuracy'] < other['Valid']['Accuracy'])
    self.metric.add_metric('Train', 'Accuracy', AccuracyMetric(self.fabric.device, task="multiclass", num_classes=self.num_classes))
    self.metric.add_metric('Train', 'Loss', CrossEntropyLoss())
    self.metric.add_metric('Valid', 'Accuracy', AccuracyMetric(self.fabric.device, task="multiclass", num_classes=self.num_classes))
    self.metric.add_metric('Valid', 'Loss', CrossEntropyLoss())
    self.metric.add_metric('Valid', 'Confusion', ConfusionMatrixMetric(self.fabric.device, task="multiclass", num_classes=self.num_classes))

  def train_epoch(self, epoch:int):
    self._default_backward()
    self._default_valid()
    if self.update_best_metric(self.metric):
      self.metric['Valid']['Confusion'].plot()
      self.save_model()

  def log_epoch(self, epoch:int):
    print(f'Epoch: {epoch}\n{self.metric}\n')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Training code for gesture models")
  add_argument(parser)
  args = parser.parse_args()

  IMUGestureTrainer(**vars(args)).train()
  # python train_imu_gesture.py --num-classes=5 --dataset-path=./local_dataset/touch_dataset