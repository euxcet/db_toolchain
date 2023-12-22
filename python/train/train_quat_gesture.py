import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics.classification import Accuracy
from torch.utils.data import DataLoader, random_split

from model.quat_gesture_model import FullyConnectedModel
from train.quat_gesture_dataset import get_quat_gesture_dataset
from lightning.fabric import Fabric, seed_everything

from train.trainer import Trainer
from train.metric import Metric, CrossEntropyLoss

class QuatGestureTrainer(Trainer):
  def __init__(self, args):
    # TODO
    pass

  def init_dataset(self):
    self.dataset = get_quat_gesture_dataset(self.args.dataset_path)
    self.train_dataset, self.valid_dataset = random_split(self.dataset, [self.args.train_ratio, 1 - self.args.train_ratio])

    self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
    self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.args.batch_size, shuffle=False)
    self.train_loader, self.valid_loader = self.fabric.setup_dataloaders(self.train_loader, self.valid_loader)

  def init_model(self):
    self.model = FullyConnectedModel(num_classes=self.args.num_classes)
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
    self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
    self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5, 8], gamma=self.args.gamma)
    self.criterion = nn.CrossEntropyLoss()

  def init_metric(self):
    self.metric = Metric()
    self.metric.add_metric('valid', 'Accuracy', Accuracy(task="multiclass", num_classes=args.num_classes).to(self.fabric.device))
    self.metric.add_metric('valid', 'Loss', CrossEntropyLoss())
    self.metric.set_compare(lambda x,y: )

  def train_epoch(self, epoch:int):
    self.model.train()
    for batch_id, (data, target) in enumerate(self.train_loader):
      self.optimizer.zero_grad()
      output = self.model(data)
      loss = self.criterion(output, target)
      self.fabric.backward(loss)
      self.optimizer.step()
    self.scheduler.step()

    self.model.eval()
    self.metric.reset(group='valid')
    with torch.no_grad():
      for data, target in self.valid_loader:
        output = self.model(data)
        self.metric.update(output, target, group='valid')
    self.metric.compute(group='valid')
    print(f'\nEpoch: {epoch}    Validation[ Average loss: {valid_loss:.4f}, Accuracy: {100 * valid_acc_value:.1f}% ]')
    if valid_acc_value > best_acc:
      best_acc = valid_acc_value
      torch.save(self.model.state_dict(), args.save_model_path)

def train(args):
  fabric = Fabric(accelerator="auto")
  seed_everything(args.seed)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Training code for gesture models")
  parser.add_argument("--batch-size", type=int, default=32, metavar="N", help="input batch size for training (default: 32)")
  parser.add_argument("--epochs", type=int, default=50, metavar="N", help="number of epochs to train (default: 50)")
  parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 1.0)")
  parser.add_argument("--gamma", type=float, default=0.1, metavar="M", help="learning rate step gamma (default: 0.7)")
  parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
  parser.add_argument("--train-ratio", type=float, default=0.8, metavar="R", help="ratio of training dataset size")
  parser.add_argument("--dataset-path", type=str, metavar="PATH", default='./local_dataset/trajectory/touch_dataset', help="dataset location")
  parser.add_argument("--num-classes", type=int, metavar="N", required=True, help="number of classes")
  parser.add_argument("--save-model-path", type=str, metavar="PATH", default="imu_gesture.pth", help="save path of the output model")
  args = parser.parse_args()

  train(args)