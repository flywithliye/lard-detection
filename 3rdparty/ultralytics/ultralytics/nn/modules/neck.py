import torch
from torch import nn
from .conv import Conv
import torch.nn.functional as F


# BiFPN
# EfficientDet: Scalable and Efficient Object Detection
# https://arxiv.org/abs/1911.09070
# BiFPN_Concat
class BiFPN_Concat2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)


class BiFPN_Concat3(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat3, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        return torch.cat(x, self.d)


# BiFPN_Add
class BiFPN_Add2(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Add2, self).__init__()
        self.w = nn.Parameter(torch.ones(
            2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.silu = nn.SiLU()

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1]))


class BiFPN_Add3(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Add3, self).__init__()
        self.w = nn.Parameter(torch.ones(
            3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.silu = nn.SiLU()

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))
    

class Upsample(nn.Module):
    """Applies convolution followed by upsampling."""

    def __init__(self, c1, c2, scale_factor=2):
        super().__init__()
        # self.cv1 = Conv(c1, c2, 1)
        # self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')  # or model='bilinear' non-deterministic
        if scale_factor == 2:
            self.cv1 = nn.ConvTranspose2d(c1, c2, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        elif scale_factor == 4:
            self.cv1 = nn.ConvTranspose2d(c1, c2, 4, 4, 0, bias=True)  # nn.Upsample(scale_factor=4, mode='nearest')
        elif scale_factor == 8:
            self.cv1 = nn.ConvTranspose2d(c1, c2, 8, 8, 0, bias=True)
            
    def forward(self, x):
        # return self.upsample(self.cv1(x))
        return self.cv1(x)


class ASFF2(nn.Module):
    """ASFF2 module for YOLO AFPN head https://arxiv.org/abs/2306.15988"""

    def __init__(self, c1, c2, level=0):
        super().__init__()
        c1_l, c1_h = c1[0], c1[1]
        self.level = level
        self.dim = c1_l, c1_h
        self.inter_dim = self.dim[self.level]
        compress_c = 8

        if level == 0:
            self.stride_level_1 = Upsample(c1_h, self.inter_dim)
        if level == 1:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 2, 2, 0)  # downsample 2x

        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weights_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, x):
        x_level_0, x_level_1 = x[0], x[1]

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
        elif self.level == 1:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = x_level_1

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v), 1)
        levels_weight = self.weights_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1] + level_1_resized * levels_weight[:, 1:2]
        return self.conv(fused_out_reduced)


class ASFF3(nn.Module):
    """ASFF3 module for YOLO AFPN head https://arxiv.org/abs/2306.15988"""

    def __init__(self, c1, c2, level=0):
        super().__init__()
        c1_l, c1_m, c1_h = c1[0], c1[1], c1[2]
        self.level = level
        self.dim = c1_l, c1_m, c1_h
        self.inter_dim = self.dim[self.level]
        compress_c = 8

        if level == 0:
            self.stride_level_1 = Upsample(c1_m, self.inter_dim)
            self.stride_level_2 = Upsample(c1_h, self.inter_dim, scale_factor=4)

        if level == 1:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 2, 2, 0)  # downsample 2x
            self.stride_level_2 = Upsample(c1_h, self.inter_dim)

        if level == 2:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 4, 4, 0)  # downsample 4x
            self.stride_level_1 = Conv(c1_m, self.inter_dim, 2, 2, 0)  # downsample 2x

        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weights_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, x):
        x_level_0, x_level_1, x_level_2 = x[0], x[1], x[2]

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = self.stride_level_2(x_level_2)

        elif self.level == 1:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)

        elif self.level == 2:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        w = self.weights_levels(levels_weight_v)
        w = F.softmax(w, dim=1)

        fused_out_reduced = level_0_resized * w[:, :1] + level_1_resized * w[:, 1:2] + level_2_resized * w[:, 2:]
        return self.conv(fused_out_reduced)
    

class ASFF4(nn.Module):
    """ASFF4 module for YOLO AFPN head, extending to four feature levels."""

    def __init__(self, c1, c2, level=0):
        super().__init__()
        # c1 is expected to be a tuple with four elements
        c1_l, c1_m, c1_h, c1_xh = c1[0], c1[1], c1[2], c1[3]
        self.level = level
        self.dim = c1_l, c1_m, c1_h, c1_xh
        self.inter_dim = self.dim[self.level]
        compress_c = 8

        # Create appropriate upsample/downsample layers for each level
        if level == 0:
            self.stride_level_1 = Upsample(c1_m, self.inter_dim)
            self.stride_level_2 = Upsample(c1_h, self.inter_dim, scale_factor=4)
            self.stride_level_3 = Upsample(c1_xh, self.inter_dim, scale_factor=8)

        if level == 1:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 2, 2, 0)
            self.stride_level_2 = Upsample(c1_h, self.inter_dim)
            self.stride_level_3 = Upsample(c1_xh, self.inter_dim, scale_factor=4)

        if level == 2:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 4, 4, 0)
            self.stride_level_1 = Conv(c1_m, self.inter_dim, 2, 2, 0)
            self.stride_level_3 = Upsample(c1_xh, self.inter_dim)

        if level == 3:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 8, 8, 0)
            self.stride_level_1 = Conv(c1_m, self.inter_dim, 4, 4, 0)
            self.stride_level_2 = Conv(c1_h, self.inter_dim, 2, 2, 0)

        # Create weight layers for each level
        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = Conv(self.inter_dim, compress_c, 1, 1)

        # Create weights to combine the levels
        self.weights_levels = nn.Conv2d(compress_c * 4, 4, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, x):
        x_level_0, x_level_1, x_level_2, x_level_3 = x[0], x[1], x[2], x[3]

        # Resize each level according to the current level
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = self.stride_level_2(x_level_2)
            level_3_resized = self.stride_level_3(x_level_3)

        elif self.level == 1:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
            level_3_resized = self.stride_level_3(x_level_3)

        elif self.level == 2:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = x_level_2
            level_3_resized = self.stride_level_3(x_level_3)

        elif self.level == 3:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = self.stride_level_2(x_level_2)
            level_3_resized = x_level_3

        # Generate weights for each level
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        level_3_weight_v = self.weight_level_3(level_3_resized)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        w = self.weights_levels(levels_weight_v)
        w = F.softmax(w, dim=1)

        # Fuse the outputs
        fused_out_reduced = (level_0_resized * w[:, :1] +
                             level_1_resized * w[:, 1:2] +
                             level_2_resized * w[:, 2:3] +
                             level_3_resized * w[:, 3:])
        return self.conv(fused_out_reduced)
    

if __name__ == '__main__':

    # python -m nn.modules.neck

    # 测试BiFPN
    x1 = torch.randn(1, 64, 32, 32)
    x2 = torch.randn(1, 64, 32, 32)
    x3 = torch.randn(1, 64, 32, 32)

    bifpn_add2 = BiFPN_Add2(64, 64)
    bifpn_add3 = BiFPN_Add3(64, 64)
    output2 = bifpn_add2([x1, x2])
    output3 = bifpn_add3([x1, x2, x3])

    print("Output2 shape:", output2.shape)
    print("Output3 shape:", output3.shape)

    bifpn_concat2 = BiFPN_Concat2(dimension=1)
    bifpn_concat3 = BiFPN_Concat3(dimension=1)

    output2 = bifpn_concat2([x1, x2])
    output3 = bifpn_concat3([x1, x2, x3])
    print("Output shape from BiFPN_Concat2:", output2.shape)
    print("Output shape from BiFPN_Concat3:", output3.shape)

    # 测试AFPN
    input_level_0 = torch.rand(1, 256, 16, 16)
    input_level_1 = torch.rand(1, 512, 8, 8)
    input_level_2 = torch.rand(1, 1024, 4, 4)

    asff2 = ASFF2([256, 512], None, level=0)
    asff3 = ASFF3([256, 512, 1024], None, level=0)

    output = asff2([input_level_0, input_level_1])
    print("ASFF2 output shape:", output.shape)
    output = asff3([input_level_0, input_level_1, input_level_2])
    print("ASFF3 output shape:", output.shape)

    # 测试AFPN
    x_level_0 = torch.randn(1, 128, 160, 160)  # Feature map from level 0: 160x160 with 128 channels
    x_level_1 = torch.randn(1, 256, 80, 80)    # Feature map from level 1: 80x80 with 256 channels
    x_level_2 = torch.randn(1, 512, 40, 40)    # Feature map from level 2: 40x40 with 512 channels
    x_level_3 = torch.randn(1, 1024, 20, 20)   # Feature map from level 3: 20x20 with 1024 channels

    # Channel dimensions for each level
    c1 = (128, 256, 512, 1024)

    # Test for each level
    for level in range(4):
        asff4 = ASFF4(c1, None, level=level)
        output = asff4((x_level_0, x_level_1, x_level_2, x_level_3))
        print(f"ASFF4 Output shape at level {level}:", output.shape)

    import torch
    import torch.nn as nn
    import torch.optim as optim

    # 假设 ASFF4 和其他必要的类（例如 Conv 和 Upsample）已经定义

    class TestNet(nn.Module):
        def __init__(self, channels):
            super(TestNet, self).__init__()
            self.asff4 = ASFF4(channels, None, level=0)  # 选择适合的level

        def forward(self, x):
            return self.asff4(x)

    # 创建模型实例
    channels = (128, 256, 512, 1024)
    model = TestNet(channels)

    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 定义损失函数
    criterion = nn.MSELoss()

    # 设置训练迭代次数
    epochs = 5

    for epoch in range(epochs):
        # 创建一些虚拟数据
        x_level_0 = torch.randn(1, 128, 160, 160)  
        x_level_1 = torch.randn(1, 256, 80, 80)    
        x_level_2 = torch.randn(1, 512, 40, 40)    
        x_level_3 = torch.randn(1, 1024, 20, 20)   
        inputs = (x_level_0, x_level_1, x_level_2, x_level_3)

        # 假设标签是相同的尺寸，这里仅为测试
        labels = torch.randn(1, 128, 160, 160)

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")