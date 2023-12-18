import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..'))

import random
import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import sklearn.metrics as metrics
from torchmetrics.classification import Accuracy
from torch.utils.data import DataLoader, random_split

from model.imu_gesture_model import GestureNetCNN
from dataset.dataset import get_gesture_dataset
from lightning.fabric import Fabric, seed_everything

def train(args):
  fabric = Fabric(accelerator="mps")
  seed_everything(args.seed)

  dataset = get_gesture_dataset(args.dataset_path)
  train_ratio = 0.8
  train_dataset, valid_dataset = random_split(dataset, [train_ratio, 1 - train_ratio])

  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
  valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
  train_loader, valid_loader = fabric.setup_dataloaders(train_loader, valid_loader)

  model = GestureNetCNN(num_classes=args.num_classes)
  optimizer = optim.Adam(model.parameters(), lr=args.lr)
  model, optimizer = fabric.setup(model, optimizer)
  # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)

  criterion = nn.CrossEntropyLoss()
  valid_acc = Accuracy(task="multiclass", num_classes=args.num_classes).to(fabric.device)
  best_acc = 0

  for epoch in range(1, args.epochs + 1):
    # loop = tqdm()
    model.train()
    for batch_id, (data, target) in enumerate(train_loader):
      optimizer.zero_grad()
      output = model(data)
      loss = criterion(output, target)
      print(output)
      fabric.backward(loss)
      optimizer.step()
    scheduler.step()

    model.eval()
    valid_loss = 0
    with torch.no_grad():
      for data, target in valid_loader:
        output = model(data)
        valid_loss += F.cross_entropy(output, target, reduction="sum").item()
        valid_acc(output, target)
    valid_loss = fabric.all_gather(valid_loss).sum() / len(valid_loader.dataset)
    valid_acc_value = valid_acc.compute()
    valid_acc.reset()
    print(f'\nValidation: Average loss: {valid_loss:.4f}, Accuracy: {100 * valid_acc_value:.1f}%\n')

    # print('Epoch: [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'
    #       .format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))
    
    if valid_acc_value > best_acc:
      best_acc = valid_acc_value
      torch.save(model.state_dict(), args.save_model_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Training code for gesture models")
  parser.add_argument("--batch-size", type=int, default=32, metavar="N", help="input batch size for training (default: 32)")
  parser.add_argument("--epochs", type=int, default=50, metavar="N", help="number of epochs to train (default: 50)")
  parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
  parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="learning rate step gamma (default: 0.7)")
  parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
  parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
  parser.add_argument("--log-interval", type=int, default=10, metavar="N", help="how many batches to wait before logging training status")
  parser.add_argument("--save-model", action="store_true", default=False, help="for saving the current Model")
  parser.add_argument("--train-ratio", type=float, default=0.8, help="ratio of training dataset size")
  parser.add_argument("--dataset-path", default='./local_dataset/trajectory/touch_dataset', help="dataset location")
  parser.add_argument("--num-classes", required=True, type=int, help="number of classes")
  parser.add_argument("--save-model-path", default="gesture.pth", help="save path of the output model")
  args = parser.parse_args()

  train(args)