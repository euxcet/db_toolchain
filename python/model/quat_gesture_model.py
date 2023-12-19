from torch import nn

class FullyConnectedModel(nn.Module):
  def __init__(self, num_classes):
    super(FullyConnectedModel, self).__init__()
    self.fc0 = nn.Linear(64, 256)
    self.relu0 = nn.ReLU()
    self.fc1 = nn.Linear(256, 128)
    self.relu1 = nn.ReLU()
    self.fc2 = nn.Linear(128, 32)
    self.relu2 = nn.ReLU()
    self.fc3 = nn.Linear(32, num_classes)

  def forward(self, x):
    x = self.fc0(x)
    x = self.relu0(x)
    x = self.fc1(x)
    x = self.relu1(x)
    x = self.fc2(x)
    x = self.relu2(x)
    x = self.fc3(x)
    return x
