import torch
import torch.nn as nn

class TrajectoryLSTMModel(nn.Module):
  def __init__(self):
    super(TrajectoryLSTMModel, self).__init__()
    input_size = 6
    hidden_size = 32
    output_size = 2
    self.time_steps = 20
    self.lstm1 = nn.LSTM(input_size, hidden_size, 1)
    self.lstm2 = nn.LSTM(hidden_size, hidden_size, 1)
    self.dropout1 = nn.Dropout(0.25)
    self.dropout2 = nn.Dropout(0.25)
    self.bn1 = nn.BatchNorm1d(num_features=128)
    self.bn2 = nn.BatchNorm1d(num_features= output_size)
    self.fc1 = nn.Linear(hidden_size * self.time_steps, 128)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(128, output_size)

  def forward(self, x):
    x = x[:, -self.time_steps:, :]
    x, (hn, cn) = self.lstm1(x)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.dropout2(x)
    x = self.fc1(x)
    x = self.relu1(x)
    x = self.fc2(x)
    return x