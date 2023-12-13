import torch
from torch import nn
import torch.nn.functional as F


class SE(nn.Module):
    # https://github.com/moskomule/senet.pytorch
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ECA(nn.Module):
    # https://blog.csdn.net/qq_43456016/article/details/132172185
    def __init__(self, in_channels, gamma=2, b=1):
        super(ECA, self).__init__()
        self.in_channels = in_channels
        self.fgp = nn.AdaptiveAvgPool2d((1, 1))
        in_channels_tensor = torch.tensor(
            self.in_channels, dtype=torch.float32)
        kernel_size = int(abs((torch.log2(in_channels_tensor) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.con1 = nn.Conv1d(1,
                              1,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.act1 = nn.Sigmoid()

    def forward(self, x):
        output = self.fgp(x)  # [B, C, W, H] -> [B, C, 1, 1]
        # [B, C, 1, 1] -> [B, C, 1] -> [B, 1, C]
        output = output.squeeze(-1).transpose(-1, -2)
        # 【B, 1, C] -> [B, C, 1] -> [B, C, 1, 1]
        output = self.con1(output).transpose(-1, -2).unsqueeze(-1)
        output = self.act1(output)  # [B, C, 1, 1]
        output = torch.multiply(x, output)
        return output


class eSE(nn.Module):
    # https://github.com/youngwanLEE/CenterMask/blob/master/maskrcnn_benchmark/modeling/backbone/vovnet.py
    def __init__(self, channel, reduction=4):
        super(eSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel, channel, kernel_size=1, padding=0)
        self.hsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)  # [B, C, W, H] -> [B, C, 1, 1]
        x = self.fc(x)  # [B, C, 1, 1] -> [B, C, 1, 1]
        x = self.hsigmoid(x)  # [B, C, 1, 1]
        return input * x


class GAM(nn.Module):
    # https://github.com/northBeggar/Plug-and-Play/blob/main/00-GAM/GAM%20attention.py
    def __init__(self, in_channels, rate=4):
        super(GAM, self).__init__()
        out_channels = in_channels
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate),
                      kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels,
                      kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)

        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out


class SA(nn.Module):
    # https://github.com/wofmanaf/SA-Net/blob/main/models/sa_resnet.py

    def __init__(self, channel, groups=64):
        super(SA, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = nn.Parameter(
            torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = nn.Parameter(
            torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups),
                               channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out


class ChannelAttention(nn.Module):
    # https://github.com/Cuthbert-Huang/CPCANet/blob/main/CPCANet/nnunet/network_architecture/CPCANet.py
    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels,
                             out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons,
                             out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return x


class CPCA(nn.Module):
    # https://github.com/Cuthbert-Huang/CPCANet/blob/main/CPCANet/nnunet/network_architecture/CPCANet.py#L582
    def __init__(self, in_channels, out_channels,
                 channelAttention_reduce=4):
        super().__init__()

        self.C = in_channels
        self.O = out_channels

        assert in_channels == out_channels
        self.ca = ChannelAttention(
            input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(
            in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels, in_channels, kernel_size=(
            1, 7), padding=(0, 3), groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(
            7, 1), padding=(3, 0), groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels, in_channels, kernel_size=(
            1, 11), padding=(0, 5), groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(
            11, 1), padding=(5, 0), groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels, in_channels, kernel_size=(
            1, 21), padding=(0, 10), groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(
            21, 1), padding=(10, 0), groups=in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels,
                              kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):

        inputs = self.conv(inputs)
        inputs = self.act(inputs)

        channel_att_vec = self.ca(inputs)
        inputs = channel_att_vec * inputs

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        return out


if __name__ == '__main__':

    input = torch.randn(1, 64, 32, 32)

    se = SE(64, reduction=16)
    output = se(input)
    print(f"SE: {output.shape}")

    eca = ECA(64, gamma=2, b=1)
    output = eca(input)
    print(f"ECA: {output.shape}")

    ese = eSE(64, reduction=4)
    output = ese(input)
    print(f"{output.shape}")

    gam = GAM(64, rate=4)
    output = gam(input)
    print(f"GAM: {output.shape}")

    sa = SA(64, groups=2)
    output = sa(input)
    print(f"SA: {output.shape}")

    cpca = CPCA(64, 64)
    output = cpca(input)
    print(f"CPCA: {output.shape}")
