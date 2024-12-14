import torch
from torch import nn
import torch.nn.functional as F

from .conv import Conv, CBAM
from .block import Bottleneck


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
    def __init__(self, in_channels, channelAttention_reduce=4):
        super().__init__()

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


class EMA(nn.Module):
    # https://github.com/YOLOonMe/EMA-attention-module/blob/main/EMA_attention_module
    def __init__(self, channels, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups,
                               channels // self.groups)
        self.conv1x1 = nn.Conv2d(
            channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(
            channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # [b*g, c//g, h, w]

        # path 1
        x_h = self.pool_h(group_x)  # [b*g, c//g, h, 1]
        # b*g, c//g, 1, w ->  [b*g, c//g, w, 1]
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))  # [b*g, c//g, h+w, 1]
        # [b*g, c//g, h, 1], [b*g, c//g, w, 1]
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        # re-weight
        x1 = self.gn(group_x * x_h.sigmoid() *
                     x_w.permute(0, 1, 3, 2).sigmoid())  # [b*g, c//g, h, w]

        # path 2
        x2 = self.conv3x3(group_x)  # [b*g, c//g, h, w]
        x11 = self.softmax(self.agp(x1).reshape(  # [b*g, c//g, h, w]->[b*g, c//g, 1, 1]->[b*g, c//g, 1]->[b*g, 1, c//g]
            b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c //
                         self.groups, -1)  # [b*g, c//g, hw]
        x21 = self.softmax(self.agp(x2).reshape(  # # [b*g, c//g, h, w]->[b*g, c//g, 1, 1]->[b*g, c//g, 1]->[b*g, 1, c//g]
            b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c //
                         self.groups, -1)  # [b*g, c//g, hw]
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)  # [b*g, 1, hw]+[b*g, 1, hw]=[b*g, 1, hw]->[b*g, 1, h, w]
                   ).reshape(b * self.groups, 1, h, w)

        # re-weight
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TA(nn.Module):
    # https://github.com/landskape-ai/triplet-attention/blob/master/triplet_attention.py
    def __init__(self, no_spatial=False):
        super(TA, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out


class LSK(nn.Module):
    # https://github.com/zcablii/LSKNet/blob/main/mmrotate/models/backbones/lsknet.py
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim//2, dim, 1)

    def forward(self, x):   
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn


class LSKA(nn.Module):
    # https://github.com/StevenLauHKHK/Large-Separable-Kernel-Attention/blob/main/models/van.py
    def __init__(self, dim, k_size):
        super().__init__()

        self.k_size = k_size

        if k_size == 7:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(
                1, 3), stride=(1, 1), padding=(0, (3-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(
                3, 1), stride=(1, 1), padding=((3-1)//2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(
                1, 3), stride=(1, 1), padding=(0, 2), groups=dim, dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(
                3, 1), stride=(1, 1), padding=(2, 0), groups=dim, dilation=2)
        elif k_size == 11:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(
                1, 3), stride=(1, 1), padding=(0, (3-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(
                3, 1), stride=(1, 1), padding=((3-1)//2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(
                1, 5), stride=(1, 1), padding=(0, 4), groups=dim, dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(
                5, 1), stride=(1, 1), padding=(4, 0), groups=dim, dilation=2)
        elif k_size == 23:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(
                1, 5), stride=(1, 1), padding=(0, (5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(
                5, 1), stride=(1, 1), padding=((5-1)//2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(
                1, 7), stride=(1, 1), padding=(0, 9), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(
                7, 1), stride=(1, 1), padding=(9, 0), groups=dim, dilation=3)
        elif k_size == 35:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(
                1, 5), stride=(1, 1), padding=(0, (5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(
                5, 1), stride=(1, 1), padding=((5-1)//2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(
                1, 11), stride=(1, 1), padding=(0, 15), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(
                11, 1), stride=(1, 1), padding=(15, 0), groups=dim, dilation=3)
        elif k_size == 41:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(
                1, 5), stride=(1, 1), padding=(0, (5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(
                5, 1), stride=(1, 1), padding=((5-1)//2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(
                1, 13), stride=(1, 1), padding=(0, 18), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(
                13, 1), stride=(1, 1), padding=(18, 0), groups=dim, dilation=3)
        elif k_size == 53:
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(
                1, 5), stride=(1, 1), padding=(0, (5-1)//2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(
                5, 1), stride=(1, 1), padding=((5-1)//2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(
                1, 17), stride=(1, 1), padding=(0, 24), groups=dim, dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(
                17, 1), stride=(1, 1), padding=(24, 0), groups=dim, dilation=3)

        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return u * attn
    
    
class C2f_SE(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions, with ATT."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.att = SE(c2)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))
    

class C2f_CBAM(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions, with ATT."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.att = CBAM(c2)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))
    

class C2f_ECA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions, with ATT."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.att = ECA(c2)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))
    

class C2f_eSE(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions, with ATT."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.att = eSE(c2)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))
    

class C2f_GAM(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions, with ATT."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.att = GAM(c2)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))
    

class C2f_SA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions, with ATT."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.att = SA(c2)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))
    

class C2f_CPCA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions, with ATT."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.att = CPCA(c2)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))
    

class C2f_EMA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions, with ATT."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.att = EMA(c2)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))
    

class C2f_TA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions, with ATT."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.att = TA()

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))
    

class C2f_LSK(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions, with ATT."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.att = LSK(c2)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))
    

class C2f_LSKA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions, with ATT."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.att = LSKA(c2)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))


if __name__ == '__main__':

    x = torch.randn(1, 64, 32, 32)
    print(x.device)

    se = SE(64, reduction=16)
    y = se(x)
    print(f"SE: {y.shape}")

    eca = ECA(64, gamma=2, b=1)
    y = eca(x)
    print(f"ECA: {y.shape}")

    ese = eSE(64, reduction=4)
    y = ese(x)
    print(f"eSE: {y.shape}")

    gam = GAM(64, rate=4)
    y = gam(x)
    print(f"GAM: {y.shape}")

    sa = SA(64, groups=2)
    y = sa(x)
    print(f"SA: {y.shape}")

    cpca = CPCA(64, channelAttention_reduce=4)
    y = cpca(x)
    print(f"CPCA: {y.shape}")

    ema = EMA(64)
    y = ema(x)
    print(f"EMA: {y.shape}")

    ta = TA()
    y = ta(x)
    print(f"TA: {y.shape}")

    lsk = LSK(64)
    y = lsk(x)
    print(f"LSK: {y.shape}")

    lska = LSKA(64, k_size=53)
    y = lska(x)
    print(f"LSKA: {y.shape}")
