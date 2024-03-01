import argparse
from typing_extensions import override
from torch import Tensor
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import os
import torch
from db_train.trainer import Trainer, TrainerParameter, add_argument
from db_train.metric import MetricGroup, CrossEntropyLoss, AccuracyMetric, ConfusionMatrixMetric
from imu_gesture_model import GestureNetCNN
from imu_gesture_dataset_numpy import IMUGestureDatasetNumpy

class DynamicGestureTrainer(Trainer):
  def __init__(self, parameter: TrainerParameter):
    super(DynamicGestureTrainer, self).__init__(parameter=parameter)
    torch.set_float32_matmul_precision('high')

  @override
  def prepare_dataset(self):
    train_x_path = os.path.join(self.parameter.dataset_path, 'train_x.npy')
    train_y_path = os.path.join(self.parameter.dataset_path, 'train_y.npy')
    test_x_path = os.path.join(self.parameter.dataset_path, 'test_x.npy')
    test_y_path = os.path.join(self.parameter.dataset_path, 'test_y.npy')
    self.train_dataset = IMUGestureDatasetNumpy(train_x_path, train_y_path).load()
    self.valid_dataset = IMUGestureDatasetNumpy(test_x_path, test_y_path).load()
    self.dataset = self.train_dataset # not used
    self.train_loader = DataLoader(self.train_dataset, batch_size=self.parameter.batch_size, shuffle=True)
    self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.parameter.batch_size, shuffle=False)
    self.train_loader, self.valid_loader = self.fabric.setup_dataloaders(self.train_loader, self.valid_loader)

  @override
  def prepare_metric(self):
    self.metric = MetricGroup()
    self.metric.add_metric('Train', 'Accuracy', AccuracyMetric(self.fabric.device, task="multiclass", num_classes=self.parameter.num_classes))
    self.metric.add_metric('Train', 'Loss', CrossEntropyLoss())
    self.metric.add_metric('Valid', 'Accuracy', AccuracyMetric(self.fabric.device, task="multiclass", num_classes=self.parameter.num_classes))
    self.metric.add_metric('Valid', 'Loss', CrossEntropyLoss())
    # self.metric.add_metric('Valid', 'Confusion', ConfusionMatrixMetric(self.fabric.device, task="multiclass", num_classes=self.parameter.num_classes))

  @override
  def prepare_optimizers(self):
    self.model = GestureNetCNN(num_classes=self.parameter.num_classes)
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.parameter.lr)
    self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
    self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5, 15], gamma=self.parameter.gamma)
    self.criterion = nn.CrossEntropyLoss()

  @override
  def training_step(self, batch: tuple) -> Tensor:
    data, target = batch
    output = self.model(data)
    loss = self.criterion(output, target)
    return loss, (output, target)

  @override
  def training_epoch(self, epoch: int):
    self.backward()
    self.validate()
    if self.update_metric():
      self.save_model()

  @override
  def log_epoch(self, epoch: int):
    if self.wandb_logger:
      self.fabric.log_dict(self.metric.to_dict(), step=epoch)
    print(f'Epoch: {epoch}\n{self.metric}\n')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Training code for gesture models")
  add_argument(parser)
  args = parser.parse_args()

  DynamicGestureTrainer(TrainerParameter.from_dict(vars(args))).train()
