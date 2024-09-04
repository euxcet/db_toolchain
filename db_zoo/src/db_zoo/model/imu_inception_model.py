import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True
        )
        # 修改为一个适用于每个时间步的全连接层
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.dropout = nn.Dropout(0.2)

        self.relu = nn.ReLU()

    def forward(self, x, v=None):
        # x:(b, n, 9) v:(b, 1, 2)
        # LSTM layer
        # v_expand = v.expand(-1, x.size(1), -1)
        # x = torch.cat([x, v_expand], dim=-1)
        output, hx = self.lstm(x, hx=None)
        # 应用全连接层到每个时间步
        # 先使用 ReLU 激活函数
        output = self.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.fc2(output)
        return output


class CellLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(CellLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 创建多个 LSTMCell
        self.lstm_cells = nn.ModuleList(
            [
                nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
                for i in range(num_layers)
            ]
        )

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()

        # 如果没有提供隐藏状态和细胞状态，则初始化为零
        if hidden is None:
            h = [
                torch.zeros(batch_size, self.hidden_size, device=x.device)
                for _ in range(self.num_layers)
            ]
            c = [
                torch.zeros(batch_size, self.hidden_size, device=x.device)
                for _ in range(self.num_layers)
            ]
        else:
            h, c = hidden

        outputs = []

        # 遍历序列中的每一个时间步
        for t in range(seq_len):
            input_t = x[:, t, :]

            # 通过每一层的 LSTMCell
            for i in range(self.num_layers):
                h[i], c[i] = self.lstm_cells[i](input_t, (h[i], c[i]))
                input_t = h[i]  # 下一层的输入是当前层的隐藏状态

            outputs.append(h[-1].unsqueeze(1))  # 记录最后一层的隐藏状态

        # 将输出的列表拼接成张量
        outputs = torch.cat(outputs, dim=1)

        return outputs, (h, c)


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
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
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
        self.lstm = nn.LSTM(
            num_channels[-1], hidden_size, batch_first=True, num_layers=1
        )
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x, hx=None, action=None, people=None):
        # x:(b, n, 6), action:(b)
        # TCN layer
        output = self.tcn(x.transpose(1, 2)).transpose(
            1, 2
        )  # (b, n-xxx, num_channels[-1])
        # if action is not None:
        #     action = action.unsqueeze(1).unsqueeze(1).expand(-1, output.size(1), -1)
        #     output = torch.cat([output, action], dim=-1)
        # LSTM layer
        output, next_hx = self.lstm(output, hx)  # (b, n-xxx, hidden_size)
        # if action is not None:
        #     action = action.unsqueeze(1).unsqueeze(1).expand(-1, output.size(1), -1)
        #     output = torch.cat([output, action], dim=-1)
        # Fully connected layer
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        if hx is not None:
            return output, next_hx
        return output


class TCNMoELSTM(nn.Module):
    """Multi TCN as experts, use people id as gating network"""

    def __init__(
        self, input_size, hidden_size, output_size, num_channels, kernel_size, dropout, num_experts=19
    ):
        super(TCNMoELSTM, self).__init__()
        self.hidden_size = hidden_size
        self.tcns = nn.ModuleList(
            [
                TemporalConvNet(
                    input_size, num_channels, kernel_size=kernel_size, dropout=dropout
                )
                for _ in range(num_experts)
            ]
        )
        self.lstm = nn.LSTM(
            num_channels[-1], hidden_size, batch_first=True, num_layers=1
        )
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x, hx=None, action=None, people=None):
        # x:(b, n, 6), action:(b), people:(b)
        # TCN layer
        outputs = []
        for tcn in self.tcns:
            output = tcn(x.transpose(1, 2)).transpose(1, 2)
            outputs.append(output)
        output = torch.stack(outputs, dim=0)  # (num_experts, b, n-xxx, num_channels[-1])
        # Select expert
        expert_idx = people.unsqueeze(1).unsqueeze(1).unsqueeze(0).expand(1, output.size(1), output.size(2), output.size(3)).long()
        output = output.gather(0, expert_idx).squeeze(0)
        # LSTM layer
        output, next_hx = self.lstm(output, hx)  # (b, n-xxx, hidden_size)
        # Fully connected layer
        output = F.relu(self.fc1(output))
        output = self.fc2(output)
        if hx is not None:
            return output, next_hx
        return output


class TCNGateNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_channels, kernel_size, dropout):
        super(TCNGateNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.tcn = TemporalConvNet(
            input_size, num_channels, kernel_size=kernel_size, dropout=dropout
        )
        self.conv1 = nn.Conv1d(num_channels[-1], hidden_size, kernel_size=16, stride=10)
        self.fc1 = nn.Linear(hidden_size * 17, 10)
        self.fc_gate = nn.Linear(10, 2)

    def forward(self, x):
        # x:(b, 200, 6)
        # TCN layer
        output = self.tcn(x.transpose(1, 2))  # (b, 128, 176)
        # Conv layer
        output = self.conv1(output)  # (b, 32, 17)
        # Flatten
        output = output.view(output.size(0), -1)  # (b, 544)
        # Fully connected layer
        output = F.relu(self.fc1(output))
        output = self.fc_gate(output)
        return output


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


class InceptionModuleOld(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModuleOld, self).__init__()
        self.branch1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, padding="same"
        )

        self.branch2 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, padding="same"
        )

        self.branch3 = nn.Conv1d(
            in_channels, out_channels, kernel_size=5, padding="same"
        )

        self.branch4_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.branch4_conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=1, padding="same"
        )

    def forward(self, x):
        branch1 = F.relu(self.branch1(x))

        branch2 = F.relu(self.branch2(x))

        branch3 = F.relu(self.branch3(x))

        branch4 = self.branch4_pool(x)
        branch4 = F.relu(self.branch4_conv(branch4))

        outputs = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return outputs


class InceptionModule(nn.Module):
    def __init__(self, ni, nf, ks=40, bottleneck=True):
        super(InceptionModule, self).__init__()

        # 计算不同卷积核大小
        ks = [ks // (2**i) for i in range(3)]
        ks = [k if k % 2 != 0 else k - 1 for k in ks]  # 确保卷积核大小为奇数

        # 判断是否使用瓶颈层
        self.bottleneck = (
            nn.Conv1d(ni, nf, 1, bias=False) if bottleneck and ni > 1 else nn.Identity()
        )

        # 创建多个卷积层，使用不同的卷积核大小
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(nf if bottleneck else ni, nf, k, padding=k // 2, bias=False)
                for k in ks
            ]
        )

        # 最大池化后接1x1卷积
        self.maxconvpool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(ni, nf, kernel_size=1, bias=False),
        )

        # 批量归一化和激活函数
        self.bn = nn.BatchNorm1d(nf * 4)
        self.act = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        # 如果使用瓶颈层，先通过瓶颈层
        x = self.bottleneck(input_tensor)

        # 将输入通过不同卷积核的卷积层和池化层
        conv_outputs = [conv(x) for conv in self.convs]
        maxpool_output = self.maxconvpool(input_tensor)

        # 将所有的输出在通道维度上拼接
        x = torch.cat(conv_outputs + [maxpool_output], dim=1)

        # 批量归一化和激活函数
        return self.act(self.bn(x))


class InceptionTimeModel(nn.Module):
    def __init__(self, input_channels, num_classes, depth=6):
        super(InceptionTimeModel, self).__init__()
        # self.inception1 = InceptionModule(input_channels, 32)
        # self.inception2 = InceptionModule(32 * 4, 32)
        # self.inception3 = InceptionModule(32 * 4, 32)
        self.inceptions = nn.ModuleList(
            [
                InceptionModule(input_channels if d == 0 else 32 * 4, 32)
                for d in range(depth)
            ]
        )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32 * 4, num_classes)

    def forward(self, x):
        # x:(b, c, n)
        for inception in self.inceptions:
            x = inception(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
