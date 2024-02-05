import torch
import torch.nn as nn

# TODO: upgrade model

class TrajectoryLSTMModel(nn.Module):
  def __init__(self, input_size=6, hidden_size=32, output_size=2, time_steps=20):
    super(TrajectoryLSTMModel, self).__init__()
    self.lstm1 = nn.LSTM(input_size, hidden_size, 1)
    self.lstm2 = nn.LSTM(hidden_size, hidden_size, 1)
    # self.dropout1 = nn.Dropout(0.25)
    # self.dropout2 = nn.Dropout(0.25)
    self.bn1 = nn.BatchNorm1d(num_features=128)
    self.bn2 = nn.BatchNorm1d(num_features= output_size)
    self.fc1 = nn.Linear(hidden_size * time_steps, 128)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(128, 32)
    self.relu2 = nn.ReLU()
    self.fc3 = nn.Linear(32, output_size)

  def forward(self, x):
    x, (hn, cn) = self.lstm1(x)
    x, (hn, cn) = self.lstm2(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = self.relu1(x)
    x = self.fc2(x)
    x = self.relu2(x)
    x = self.fc3(x)
    return x
