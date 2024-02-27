import argparse
from typing_extensions import override
from torch import Tensor
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from db_train.trainer import Trainer, TrainerParameter, add_argument
from db_train.metric import MetricGroup, CrossEntropyLoss, AccuracyMetric, ConfusionMatrixMetric
from imu_gesture_model import GestureNetCNN
from imu_gesture_dataset import IMUGestureDataset

class DynamicGestureTrainer(Trainer):
  def __init__(self, parameter: TrainerParameter) -> None:
    super(DynamicGestureTrainer, self).__init__(parameter=parameter)

  @override
  def prepare_dataset(self) -> None:
    self.dataset = IMUGestureDataset(self.parameter.dataset_path).load()
    self.train_dataset, self.valid_dataset = random_split(self.dataset, [self.parameter.train_ratio, self.parameter.valid_ratio])
    self.train_loader = DataLoader(self.train_dataset, batch_size=self.parameter.batch_size, shuffle=True)
    self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.parameter.batch_size, shuffle=False)
    self.train_loader, self.valid_loader = self.fabric.setup_dataloaders(self.train_loader, self.valid_loader)

  @override
  def prepare_metric(self) -> None:
    self.metric = MetricGroup()
    self.metric.add_metric('Train', 'Accuracy', AccuracyMetric(self.fabric.device, task="multiclass", num_classes=self.parameter.num_classes))
    self.metric.add_metric('Train', 'Loss', CrossEntropyLoss())
    self.metric.add_metric('Valid', 'Accuracy', AccuracyMetric(self.fabric.device, task="multiclass", num_classes=self.parameter.num_classes))
    self.metric.add_metric('Valid', 'Loss', CrossEntropyLoss())
    self.metric.add_metric('Valid', 'Confusion', ConfusionMatrixMetric(self.fabric.device, task="multiclass", num_classes=self.parameter.num_classes))

  @override
  def prepare_optimizers(self) -> None:
    self.model = GestureNetCNN(num_classes=self.parameter.num_classes)
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.parameter.lr)
    self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
    self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5, 8], gamma=self.parameter.gamma)
    self.criterion = nn.CrossEntropyLoss()

  @override
  def training_step(self, batch: tuple) -> Tensor:
    data, target = batch
    output = self.model(data)
    loss = self.criterion(output, target)
    return loss, (output, target)

  @override
  def training_epoch(self, epoch: int) -> None:
    self.backward()
    self.validate()
    if self.update_metric():
      self.save_model()

  @override
  def log_epoch(self, epoch: int) -> None:
    print(f'Epoch: {epoch}\n{self.metric}\n')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Training code for gesture models")
  add_argument(parser)
  args = parser.parse_args()

  DynamicGestureTrainer(TrainerParameter.from_dict(vars(args))).train()
