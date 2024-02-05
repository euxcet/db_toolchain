import torch.nn as nn

# TODO: upgrade model

class MBConv1(nn.Module):
  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int,
      stride: int = 1,
      padding: int = 0,
  ) -> None:
    super(MBConv1, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, 1,
                stride=1, padding=0, bias=False),
      nn.BatchNorm2d(out_channels),
    )
    if in_channels != out_channels:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels)
      )
    else:
      self.shortcut = nn.Sequential()

  def forward(self, x):
    out = self.conv(x)
    out += self.shortcut(x)
    return out

class MBConv6(nn.Module):
  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int,
      stride: int = 1,
      padding: int = 0,
      expand_ratio: int = 6,
  ) -> None:
    super(MBConv6, self).__init__()
    self.expand_ratio = expand_ratio
    mid_channels = round(expand_ratio * in_channels)
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, mid_channels, 1,
                stride=1, padding=0, bias=False),
      nn.BatchNorm2d(mid_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(mid_channels, mid_channels, kernel_size, stride=stride,
                padding=padding, groups=mid_channels, bias=False),
      nn.BatchNorm2d(mid_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(mid_channels, out_channels, 1,
                stride=1, padding=0, bias=False),
      nn.BatchNorm2d(out_channels),
    )

  def forward(self, x):
    out = self.conv(x)
    if x.shape == out.shape:
      out += x
    return out

class ConvBlock(nn.Module):
  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int,
      stride: int,
  ) -> None:
    super(ConvBlock, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels,
                          kernel_size, stride, padding=1, bias=False)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    out = self.conv(x)
    out = self.bn(out)
    out = self.relu(out)
    return out

class CompoundBlock(nn.Module):
  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int,
      stride: int,
      num_blocks: int,
  ) -> None:
    super(CompoundBlock, self).__init__()
    layers = []
    for _ in range(num_blocks):
      layers.append(ConvBlock(in_channels, out_channels, kernel_size, stride))
      in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    self.block = nn.Sequential(*layers)

  def forward(self, x):
    out = self.block(x)
    return out

class SeparableConv(nn.Module):
  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int,
      stride: int,
  ) -> None:
    super(SeparableConv, self).__init__()
    self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                                    padding=0, groups=in_channels, bias=False)
    self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

  def forward(self, x):
    x = self.depthwise_conv(x)
    x = self.pointwise_conv(x)
    return x

class GestureNetCNN(nn.Module):
  def __init__(self, num_classes:int):
    super(GestureNetCNN, self).__init__()
    self.conv1 = MBConv1(6, 12, (1, 9), stride=1, padding=(0, 4))  
    self.conv2 = MBConv6(12, 24, (1, 10), stride=2, padding=(0, 4)) 
    self.conv3 = MBConv6(24, 24, (1, 10), stride=2, padding=(0, 4)) 
    self.conv4 = SeparableConv(24, 24, (1, 10), stride=1) 
    self.maxpool1 = nn.MaxPool1d(kernel_size=10, stride=9)
    self.flattern = nn.Flatten()

    self.linear1 = nn.Linear(96, 80)
    self.batch_norm1 = nn.BatchNorm1d(80)
    self.dropout1 = nn.Dropout(0.5)
    self.relu1 = nn.ReLU(inplace=True)

    self.linear2 = nn.Linear(80, 40)
    self.batch_norm2 = nn.BatchNorm1d(40)
    self.relu2 = nn.ReLU(inplace=True)
    self.dropout2 = nn.Dropout(0.5)

    self.linear3 = nn.Linear(40, num_classes)

  def forward(self, x):
    out = self.conv1(x)
    out = self.conv2(out)
    out = self.conv3(out)
    out = self.conv4(out)
    out = out.view(out.size(0), out.size(1), -1)
    out = self.maxpool1(out)
    out = self.flattern(out)
    out = self.linear1(out)
    out = self.relu1(out)
    out = self.linear2(out)
    out = self.relu2(out)
    out = self.linear3(out)
    return out
