import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from matplotlib import pyplot as plt
from sklearn import metrics
from model.quat_gesture_model import FullyConnectedModel

def get_dataset(users):
  all_data = []
  all_label = []
  for i in range(100):
    data = []
    for user in users:
      folder_path = os.path.join('glove_dataset', user, str(i))
      if os.path.exists(folder_path):
        for f in os.listdir(folder_path):
          if f.endswith('.npy'):
            data.append(np.load(os.path.join(folder_path, f)))
    if len(data) > 0:
      new_data = np.concatenate(data).astype('float32')
      all_label.append(np.ones(new_data.shape[0]).astype('long') * len(all_data))
      all_data.append(new_data)
  
  return len(all_data), TensorDataset(torch.tensor(np.concatenate(all_data)), torch.tensor(np.concatenate(all_label)))

def calc_metric(model, loader, device):
  y_pred = np.zeros(0)
  y_true = np.zeros(0)
  for x_val, y_val in loader:
      x_val = x_val.to(device)
      y_val = y_val.to(device)
      out = model(x_val)
      preds = F.log_softmax(out, dim=1).argmax(dim=1)
      y_pred = np.append(y_pred, preds.cpu().numpy())
      y_true = np.append(y_true, y_val.cpu().numpy())

  accuracy = metrics.accuracy_score(y_true, y_pred)
  return accuracy

def train():
    # reset random seed
    np.random.seed(0)

    num_class, train_dataset = get_dataset(['zcc', 'zcc1', 'lxy', 'lc', 'hz', 'lxy2', 'wxt'])
    num_class, valid_dataset = get_dataset(['lzj'])
    # device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    device = 'cpu'
    model = FullyConnectedModel(num_class).to(device)

    batch_size = 32

    # create train and val data loader
    print(num_class, len(train_dataset), len(valid_dataset))
    train_loader = DataLoader(train_dataset, batch_size, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size, num_workers=2)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_metric = 0.0

    print('Start training.')
    for epoch in range(500):
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        train_accuracy = calc_metric(model, train_loader, device)
        valid_accuracy = calc_metric(model, valid_loader, device)
        print(train_loss, train_accuracy, valid_accuracy)

        if valid_accuracy > best_metric:
            best_metric = valid_accuracy
            torch.save(model.state_dict(), 'glove.pth')
            print(f'\tbest model metric: {best_metric:.3f}')

    return model

if __name__ == '__main__':
   train()
