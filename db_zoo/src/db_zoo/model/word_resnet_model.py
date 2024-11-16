import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        # 修改为一个适用于每个时间步的全连接层
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)

        self.relu = nn.ReLU()

    def forward(self, x, hx=None):
        # LSTM layer
        output, hx = self.lstm(x, hx=hx)
        # 应用全连接层到每个时间步
        # 先使用 ReLU 激活函数
        output = self.relu(self.fc1(output))
        output = self.fc2(output)
        return output, hx


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 定义第一个卷积层，输入通道为 features，输出通道为 16
        self.conv1 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=(1, 3), stride=1, padding=(0, 1)
        )
        # 定义第二个卷积层
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
        )
        # 定义池化层，沿时间维度池化
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        # 定义全连接层，输出12维向量，注意这里我们不再直接展平所有数据
        # 假设历史帧经处理后的大小是7
        self.fc1 = nn.Linear(32 * 7, 12)

    def forward(self, x):
        # x的初始形状: (batch_size, n, history_frame, features)
        x = x.permute(0, 3, 1, 2)  # 调整维度顺序
        # 应用第一个卷积层和池化
        x = self.pool(
            F.relu(self.conv1(x))
        )  # 形状变为: (batch_size, 16, n, history_frame/2)
        # 应用第二个卷积层和池化
        x = self.pool(F.relu(self.conv2(x)))  # 形状变为: (batch_size, 32, n, 7)
        # x的形状还原
        x = x.permute(0, 2, 1, 3)  # 形状变为: (batch_size, n, 32, 7)
        # 保留n维度，为每个n展平后续的维度
        x = x.reshape(x.size(0), x.size(1), -1)  # 形状变为: (batch_size, n, 32 * 7)
        # 应用全连接层到每个时间步骤
        x = self.fc1(x)  # 形状变为: (batch_size, n, 12)
        return x


class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleGRU, self).__init__()
        self.hidden_size = hidden_size
        # 使用GRU替代LSTM
        self.gru = nn.GRU(input_size, hidden_size, num_layers=5, batch_first=True)
        # 全连接层
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        # ReLU激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # GRU layer
        output, hidden = self.gru(x)
        # 应用全连接层到每个时间步
        output = self.relu(self.fc1(output))
        output = self.fc2(output)
        return output


class CNNtoLSTM(nn.Module):
    def __init__(self, history_frame=30):
        super(CNNtoLSTM, self).__init__()
        self.history_frame = history_frame
        self.cnn = SimpleCNN()
        self.lstm = SimpleLSTM(input_size=12, hidden_size=256, output_size=2)

    def prepare_data(self, input_data):
        batch_size, n, features = input_data.shape  # (batch_size, n, features)

        # 使用pad函数在时间维度前方填充历史帧数-1的数据，填充内容为第一帧数据
        first_frames = (
            input_data[:, 0, :].unsqueeze(1).repeat(1, self.history_frame - 1, 1)
        )
        padded_input = torch.cat(
            [first_frames, input_data], dim=1
        )  # 前面填充历史帧-1个第一帧

        # 创建输出数据的容器
        output_data = torch.zeros(
            (batch_size, n, self.history_frame, features), device=input_data.device
        )

        # 利用高效的张量操作收集每个时间点的历史数据
        for i in range(self.history_frame):
            output_data[:, :, i, :] = padded_input[:, i : n + i, :]

        # 调整输出张量的维度以匹配 (batch_size, n, history_frame, features)
        return output_data

    def forward(self, x):
        # 准备数据
        x = self.prepare_data(x)  # (batch_size, n, history_frame, features)
        # CNN 处理
        cnn_out = self.cnn(x)  # (batch_size, n, 12)
        # LSTM 处理
        output = self.lstm(cnn_out)
        return output


class CNNtoGRU(nn.Module):
    def __init__(self, history_frame=30):
        super(CNNtoGRU, self).__init__()
        self.history_frame = history_frame
        self.cnn = SimpleCNN()
        self.gru = SimpleGRU(input_size=12, hidden_size=128, output_size=2)

    def prepare_data(self, input_data):
        batch_size, n, features = input_data.shape

        # 使用pad函数在时间维度前方填充历史帧数-1的数据，填充内容为第一帧数据
        first_frames = (
            input_data[:, 0, :].unsqueeze(1).repeat(1, self.history_frame - 1, 1)
        )
        padded_input = torch.cat(
            [first_frames, input_data], dim=1
        )  # 前面填充历史帧-1个第一帧

        # 创建输出数据的容器
        output_data = torch.zeros(
            (batch_size, n, self.history_frame, features), device=input_data.device
        )

        # 利用高效的张量操作收集每个时间点的历史数据
        for i in range(self.history_frame):
            output_data[:, :, i, :] = padded_input[:, i : n + i, :]

        # 调整输出张量的维度以匹配 (batch_size, n, history_frame, features)
        return output_data

    def forward(self, x):
        # 准备数据
        x = self.prepare_data(x)  # 结果形状为 (batch_size, n, history_frame, features)
        # CNN 处理
        cnn_out = self.cnn(x)  # (batch_size, n, 12)
        # GRU 处理
        output = self.gru(cnn_out)
        return output


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, y_pred, y_true, mask=None):
        # 计算 MSE 损失
        mse_loss = self.mse(y_pred, y_true)  # 形状为 (32, 12000, 2)
        if mask is not None:
            # 扩展 mask 的形状以匹配 y_pred 和 y_true
            mask_expanded = mask.unsqueeze(-1)  # 新形状为 (32, 12000, 1)
            mask_expanded = mask_expanded.expand_as(
                y_pred
            )  # 扩展形状匹配 (32, 12000, 2)
            mse_loss = mse_loss * mask_expanded  # 应用扩展后的掩码
            mse_loss_mean = mse_loss.sum() / mask_expanded.sum()
        else:
            # 没有掩码时计算全局平均
            mse_loss_mean = mse_loss.mean()

        # 结合两种损失
        return mse_loss_mean


class MSEWithMask(nn.Module):
    """在掩码之外计算平均 mse loss"""

    def __init__(self):
        super(MSEWithMask, self).__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, y_pred, y_true, mask=None):
        # 计算 MSE 损失
        mse_loss = self.mse(y_pred, y_true)  # 形状为 (32, 12000, 2)
        if mask is not None:
            # 扩展 mask 的形状以匹配 y_pred 和 y_true
            mask_expanded = mask.unsqueeze(-1)  # 新形状为 (32, 12000, 1)
            mask_expanded = mask_expanded.expand_as(
                y_pred
            )  # 扩展形状匹配 (32, 12000, 2)
            mse_loss = mse_loss * mask_expanded  # 应用扩展后的掩码
            mse_loss_mean = mse_loss.sum() / mask_expanded.sum()
        else:
            # 没有掩码时计算全局平均
            mse_loss_mean = mse_loss.mean()

        # 结合两种损失
        return {"loss": mse_loss_mean, "mse_loss": mse_loss_mean}


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # return x[:, :, : -self.chomp_size].contiguous()
        return x[:, :, self.chomp_size :].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class MyTemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(MyTemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.padding = padding
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = (
            x[:, :, self.padding : -self.padding]
            if self.downsample is None
            else self.downsample(x[:, :, self.padding : -self.padding])
        )
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                MyTemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LSTMTCN(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_channels, kernel_size, dropout
    ):
        super(LSTMTCN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.tcn = TemporalConvNet(
            hidden_size, num_channels, kernel_size=kernel_size, dropout=dropout
        )
        self.fc1 = nn.Linear(num_channels[-1], num_channels[-1] // 2)
        self.fc2 = nn.Linear(num_channels[-1] // 2, output_size)

    def forward(self, x):
        # x:(b, n, 9)
        # LSTM layer
        output, _ = self.lstm(x)
        # TCN layer
        output = self.tcn(output.transpose(1, 2)).transpose(1, 2)
        # Fully connected layer
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        return output


class TCNLSTM(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_channels, kernel_size, dropout
    ):
        super(TCNLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.tcn = TemporalConvNet(
            input_size, num_channels, kernel_size=kernel_size, dropout=dropout
        )
        self.lstm = nn.LSTM(num_channels[-1], hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    # def forward(self, x, hx):
    #     # x:(b, n, 9)
    #     output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
    #     output, next_hx = self.lstm(output, hx)
    #     output = F.relu(self.fc1(output))
    #     output = self.fc2(output)
    #     return output, next_hx

    def forward(self, x, hx = None):
        # x:(b, n, 9)
        # TCN layer
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        # LSTM layer
        output, next_hx = self.lstm(output, hx)
        # Fully connected layer
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        # if hx is not None:
        #     return output, next_hx
        return output, next_hx


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_prob=0.0):
        super(BasicBlock, self).__init__()
        self.dropout_prob = dropout_prob

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.dropout1 = nn.Dropout2d(self.dropout_prob)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout2 = nn.Dropout2d(self.dropout_prob)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_blocks, block=BasicBlock, num_classes=10, dropout_prob=0.3):
        super(ResNet, self).__init__()
        self.inplanes = 8
        self.dropout_fc = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(
            1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.layer1 = self._make_layer(
            block, 8, num_blocks[0], stride=1, dropout_prob=dropout_prob
        )
        self.layer2 = self._make_layer(
            block, 16, num_blocks[1], stride=1, dropout_prob=dropout_prob
        )
        self.layer3 = self._make_layer(
            block, 32, num_blocks[2], stride=1, dropout_prob=dropout_prob
        )
        self.fc = nn.Linear(32, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, dropout_prob):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride, dropout_prob))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)  # x的维度现在应为(batch_size, 1, 6, 20)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
