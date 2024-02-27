import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from typing_extensions import override
from torch.utils.data import DataLoader, random_split
from db_train.trainer import Trainer, TrainerParameter, add_argument
from db_train.metric import MetricGroup, TrajectoryLoss
from imu_trajectory_model import TrajectoryLSTMModel
from imu_trajectory_dataset import IMUTrajectoryDataset

class Loss(nn.Module):
  def __init__(self):
    super().__init__()
    self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-8)

  def forward(self, x, y):
    length_loss = torch.mean(torch.abs((torch.norm(x, dim=1) - torch.norm(y, dim=1))))
    angle_loss = 1 - torch.mean(self.cos_sim(x, y))
    return length_loss + angle_loss * 2

class TrajectoryTrainer(Trainer):
  def __init__(self, parameter: TrainerParameter):
    super(TrajectoryTrainer, self).__init__(parameter=parameter)

  @override
  def prepare_dataset(self):
    self.dataset = IMUTrajectoryDataset(self.parameter.dataset_path).load()
    self.train_dataset, self.valid_dataset = random_split(self.dataset, [self.parameter.train_ratio, self.parameter.valid_ratio])
    self.train_loader = DataLoader(self.train_dataset, batch_size=self.parameter.batch_size, shuffle=True)
    self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.parameter.batch_size, shuffle=False)
    self.train_loader, self.valid_loader = self.fabric.setup_dataloaders(self.train_loader, self.valid_loader)

  @override
  def prepare_metric(self):
    self.metric = MetricGroup(lt_func=lambda x, other: x['Valid']['Loss'] < other['Valid']['Loss'])
    self.metric.add_metric('Train', 'Loss', TrajectoryLoss())
    self.metric.add_metric('Valid', 'Loss', TrajectoryLoss())

  @override
  def prepare_optimizers(self):
    self.model = TrajectoryLSTMModel()
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.parameter.lr)
    self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8, patience=10)
    self.criterion = Loss()

  @override
  def training_step(self, batch: tuple) -> Tensor:
    data, target = batch
    output = self.model(data)
    loss = self.criterion(output, target)
    return loss, (output, target)

  @override
  def training_epoch(self, epoch: int) -> None:
    self.backward(step_scheduler=False)
    self.scheduler.step(self.metric['Train']['Loss'].value)
    self.validate()
    if self.update_metric():
      self.save_model()

  @override
  def log_epoch(self, epoch: int) -> None:
    print(f'Epoch: {epoch}\n{self.metric}\n')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Training code for trajectory models")
  add_argument(parser)
  args = parser.parse_args()

  TrajectoryTrainer(TrainerParameter.from_dict(vars(args))).train()