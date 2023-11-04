import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import sklearn.metrics as metrics
from torch.utils.data import DataLoader, random_split
from model.gesture_cnn import GestureNetCNN
from dataset import get_gesture_dataset
import argparse

torch.manual_seed(3407)

def train(args):
  batch_size = 32
  num_epochs = 200
  learning_rate = 0.01

  dataset = get_gesture_dataset('1014_dataset')

  train_ratio = 0.8
  val_ratio = 0.1
  test_ratio = 0.1

  num_data = len(dataset)
  num_train = int(train_ratio * num_data)
  num_val = int(val_ratio * num_data)
  num_test = num_data - num_train - num_val

  train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_val, num_test])

  print(len(dataset), len(train_dataset), len(val_dataset), len(test_dataset))

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
  # device = torch.device('cpu')

  ring_model = GestureNetCNN(num_classes=28).to(device)
  criterion = nn.CrossEntropyLoss()

  optimizer = optim.Adam(ring_model.parameters(), lr=learning_rate, weight_decay=5e-4)
  scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 8], gamma=0.1)

  best_acc = 0.0

  for epoch in range(num_epochs):
    ring_model.train()
    train_loss = 0.0
    train_acc = 0.0
    for ring_data, labels in train_loader:
      ring_data = ring_data.to(device)
      labels = labels.to(device)
      optimizer.zero_grad()
      outputs = ring_model(ring_data)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      train_loss += loss.item() * ring_data.size(0)
      _, predictions = torch.max(outputs, 1)
      train_acc += torch.sum(predictions == labels.data)
    train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_acc / len(train_loader.dataset)

    ring_model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
      for ring_data, labels in val_loader:
        ring_data = ring_data.to(device)
        labels = labels.to(device)
        outputs = ring_model(ring_data)
        loss = criterion(outputs, labels)
        val_loss += loss.item() * ring_data.size(0)
        _, predictions = torch.max(outputs, 1)
        val_acc += torch.sum(predictions == labels.data)
      val_loss = val_loss / len(val_loader.dataset)
      val_acc = val_acc / len(val_loader.dataset)

    print('Epoch: [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'
        .format(epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))
    
    scheduler.step()

    if val_acc > best_acc:
      best_acc = val_acc
      ring_model.to('cpu')
      torch.save(ring_model.state_dict(), 'ring.pth')
      ring_model.to('mps')

  ring_model.load_state_dict(torch.load('ring.pth'))
  ring_model.eval()
  test_loss = 0.0
  test_acc = 0.0
  predictions_list = []
  labels_list = []
  with torch.no_grad():
    for ring_data, labels in test_loader:
      ring_data = ring_data.to(device)
      labels = labels.to(device)
      outputs = ring_model(ring_data)
      loss = criterion(outputs, labels)
      test_loss += loss.item() * ring_data.size(0)
      _, predictions = torch.max(outputs, 1)
      predictions_list += predictions.cpu().numpy().tolist()
      labels_list += labels.cpu().numpy().tolist()
      test_acc += torch.sum(predictions == labels.data)
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_acc / len(test_loader.dataset)
    print('Test Loss: {:.4f}, Test Acc: {:.4f}, Test f1: {:.4f}'.format(test_loss, test_acc, metrics.f1_score(labels_list, predictions_list, average='macro')))

  conf_matrix = metrics.confusion_matrix(labels_list, predictions_list, labels=[i for i in range(18)])
  print("Confusion Matrix:")
  print(conf_matrix)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--name', type=str, required=True)
  parser.add_argument('--user', type=str, required=True)
  parser.add_argument('--action', type=str, required=True)
  parser.add_argument('--number', type=int, required=True, default=10)
  parser.add_argument('--device', type=str, required=True)
  parser.add_argument('--sample_time', type=float, required=False, default=1.0)
  parser.add_argument('--restore_time', type=float, required=False, default=1.0)
  args = parser.parse_args()
  train(args)