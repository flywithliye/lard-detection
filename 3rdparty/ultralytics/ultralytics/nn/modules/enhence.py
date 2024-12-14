import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3, kernel_size=5, channels=3):
        super().__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel(kernel_size, channels)

    def gauss_kernel(self, kernel_size, channels):
        kernel = cv2.getGaussianKernel(kernel_size, 0).dot(
            cv2.getGaussianKernel(kernel_size, 0).T)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).repeat(
            channels, 1, 1, 1)
        kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
        return kernel

    def conv_gauss(self, x, kernel):
        n_channels, _, kw, kh = kernel.shape
        x = torch.nn.functional.pad(x, (kw // 2, kh // 2, kw // 2, kh // 2),
                                    mode='reflect')  # replicate    # reflect
        x = torch.nn.functional.conv2d(x, kernel, groups=n_channels)
        return x

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def pyramid_down(self, x):
        return self.downsample(self.conv_gauss(x, self.kernel))

    def upsample(self, x):
        up = torch.zeros((x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2),
                         device=x.device)
        up[:, :, ::2, ::2] = x * 4

        return self.conv_gauss(up, self.kernel)

    def pyramid_decom(self, img):
        self.kernel = self.kernel.to(img.device)
        current = img
        pyr = []
        for _ in range(self.num_high):
            down = self.pyramid_down(current)
            up = self.upsample(down)
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[0]
        for level in pyr[1:]:
            up = self.upsample(image)
            image = up + level
        return image


# DE-Yolo
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        assert kernel_size in (3, 5, 7), "kernel size must be 3 or 5 or 7"

        self.conv = nn.Conv2d(2,
                              1,
                              kernel_size,
                              padding=kernel_size // 2,
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avgout, maxout], dim=1)
        attention = self.conv(attention)
        return self.sigmoid(attention) * x


class Trans_guide(nn.Module):
    def __init__(self, ch=16):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(6, ch, 3, padding=1),
            nn.LeakyReLU(True),
            SpatialAttention(3),
            nn.Conv2d(ch, 3, 3, padding=1),
        )

    def forward(self, x):
        return self.layer(x)


class Trans_low(nn.Module):
    def __init__(
        self,
        ch_blocks=64,
        ch_mask=16,
    ):
        super().__init__()

        self.encoder = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1),
                                     nn.LeakyReLU(True),
                                     nn.Conv2d(16, ch_blocks, 3, padding=1),
                                     nn.LeakyReLU(True))

        self.mm1 = nn.Conv2d(ch_blocks,
                             ch_blocks // 4,
                             kernel_size=1,
                             padding=0)
        self.mm2 = nn.Conv2d(ch_blocks,
                             ch_blocks // 4,
                             kernel_size=3,
                             padding=3 // 2)
        self.mm3 = nn.Conv2d(ch_blocks,
                             ch_blocks // 4,
                             kernel_size=5,
                             padding=5 // 2)
        self.mm4 = nn.Conv2d(ch_blocks,
                             ch_blocks // 4,
                             kernel_size=7,
                             padding=7 // 2)

        self.decoder = nn.Sequential(nn.Conv2d(ch_blocks, 16, 3, padding=1),
                                     nn.LeakyReLU(True),
                                     nn.Conv2d(16, 3, 3, padding=1))

        self.trans_guide = Trans_guide(ch_mask)

    def forward(self, x):
        x1 = self.encoder(x)
        x1_1 = self.mm1(x1)
        x1_2 = self.mm1(x1)
        x1_3 = self.mm1(x1)
        x1_4 = self.mm1(x1)
        x1 = torch.cat([x1_1, x1_2, x1_3, x1_4], dim=1)
        x1 = self.decoder(x1)

        out = x + x1
        out = torch.relu(out)

        mask = self.trans_guide(torch.cat([x, out], dim=1))
        return out, mask


class SFT_layer(nn.Module):
    def __init__(self, in_ch=3, inter_ch=32, out_ch=3, kernel_size=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(inter_ch, out_ch, kernel_size, padding=kernel_size // 2))
        self.shift_conv = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size, padding=kernel_size // 2))
        self.scale_conv = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size, padding=kernel_size // 2))

    def forward(self, x, guide):
        x = self.encoder(x)
        scale = self.scale_conv(guide)
        shift = self.shift_conv(guide)
        x += x * scale + shift
        x = self.decoder(x)
        return x


class Trans_high(nn.Module):
    def __init__(self, in_ch=3, inter_ch=32, out_ch=3, kernel_size=3):
        super().__init__()

        self.sft = SFT_layer(in_ch, inter_ch, out_ch, kernel_size)

    def forward(self, x, guide):
        return x + self.sft(x, guide)


class Up_guide(nn.Module):
    def __init__(self, kernel_size=1, ch=3):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(ch,
                      ch,
                      kernel_size,
                      stride=1,
                      padding=kernel_size // 2,
                      bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


class DENet(nn.Module):
    # https://github.com/NIvykk/DENet/blob/main/models/deyolo.py
    def __init__(self,
                 num_high=3,
                 ch_blocks=32,
                 up_ksize=1,
                 high_ch=32,
                 high_ksize=3,
                 ch_mask=32,
                 gauss_kernel=5):
        super().__init__()
        self.num_high = num_high
        self.lap_pyramid = Lap_Pyramid_Conv(num_high, gauss_kernel)
        self.trans_low = Trans_low(ch_blocks, ch_mask)

        for i in range(0, self.num_high):
            self.__setattr__('up_guide_layer_{}'.format(i),
                             Up_guide(up_ksize, ch=3))
            self.__setattr__('trans_high_layer_{}'.format(i),
                             Trans_high(3, high_ch, 3, high_ksize))

    def forward(self, x):
        pyrs = self.lap_pyramid.pyramid_decom(img=x)

        trans_pyrs = []
        trans_pyr, guide = self.trans_low(pyrs[-1])
        trans_pyrs.append(trans_pyr)

        commom_guide = []
        for i in range(self.num_high):
            guide = self.__getattr__('up_guide_layer_{}'.format(i))(guide)
            commom_guide.append(guide)

        for i in range(self.num_high):
            trans_pyr = self.__getattr__('trans_high_layer_{}'.format(i))(
                pyrs[-2 - i], commom_guide[i])
            trans_pyrs.append(trans_pyr)

        out = self.lap_pyramid.pyramid_recons(trans_pyrs)

        return out


# PE-Yolo
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv_x = nn.Conv2d(in_features, out_features, 3, padding=1)

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return self.conv_x(x + self.block(x))


class LowPassModule(nn.Module):
    def __init__(self, in_channel, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])
        self.relu = nn.ReLU()
        ch = in_channel // 4
        self.channel_splits = [ch, ch, ch, ch]

    def _make_stage(self, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        return nn.Sequential(prior)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        feats = torch.split(feats, self.channel_splits, dim=1)
        priors = [F.interpolate(input=self.stages[i](feats[i]), size=(h, w), mode='bilinear', align_corners=True) for i in range(4)]
        bottle = torch.cat(priors, 1)
        return self.relu(bottle)


def sobel(img):
    add_x_total = torch.zeros_like(img)

    for i in range(img.shape[0]):
        x = img[i, :, :, :].squeeze(0).cpu().numpy().transpose(1, 2, 0)

        x = x * 255
        x_x = cv2.Sobel(x, cv2.CV_64F, 1, 0)
        x_y = cv2.Sobel(x, cv2.CV_64F, 0, 1)
        add_x = cv2.addWeighted(x_x, 0.5, x_y, 0.5, 0)
        add_x = transforms.ToTensor()(add_x).unsqueeze(0)
        add_x_total[i, :, :, :] = add_x

    return add_x_total


class DPM(nn.Module):
    def __init__(self, inplanes, planes, act=nn.LeakyReLU(negative_slope=0.2, inplace=True), bias=False):
        super(DPM, self).__init__()

        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=bias),
            act,
            nn.Conv2d(planes, inplanes, kernel_size=1, bias=bias)
        )

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        x = x + channel_add_term
        return x


class AE(nn.Module):
    def __init__(self, n_feat, reduction=8, bias=False, act=nn.LeakyReLU(negative_slope=0.2, inplace=True), groups=1):
        super(AE, self).__init__()

        self.n_feat = n_feat
        self.groups = groups
        self.reduction = reduction
        self.agg = nn.Conv2d(6,
                             3,
                             1,
                             stride=1,
                             padding=0,
                             bias=False)
        self.conv_edge = nn.Conv2d(3, 3, kernel_size=1, bias=bias)

        self.res1 = ResidualBlock(3, 32)
        self.res2 = ResidualBlock(32, 3)
        self.dpm = nn.Sequential(DPM(32, 32))

        self.conv1 = nn.Conv2d(3, 32, kernel_size=1)
        self.conv2 = nn.Conv2d(32, 3, kernel_size=1)
        self.lpm = LowPassModule(32)
        self.fusion = nn.Conv2d(6, 3, kernel_size=1)

    def forward(self, x):
        s_x = sobel(x)
        s_x = self.conv_edge(s_x)

        res = self.res1(x)
        res = self.dpm(res)
        res = self.res2(res)
        out = torch.cat([res, s_x + x], dim=1)
        out = self.agg(out)

        low_fea = self.conv1(x)
        low_fea = self.lpm(low_fea)
        low_fea = self.conv2(low_fea)

        out = torch.cat([out, low_fea], dim=1)
        out = self.fusion(out)

        return out


class PENet(nn.Module):
    # https://github.com/XiangchenYin/PE-YOLO/blob/master/PENet.py
    def __init__(self,
                 num_high=3,
                 ch_blocks=32,
                 up_ksize=1,
                 high_ch=32,
                 high_ksize=3,
                 ch_mask=32,
                 gauss_kernel=5):
        super().__init__()
        self.num_high = num_high
        self.lap_pyramid = Lap_Pyramid_Conv(num_high, gauss_kernel)

        for i in range(0, self.num_high + 1):
            self.__setattr__('AE_{}'.format(i), AE(3))

    def forward(self, x):
        pyrs = self.lap_pyramid.pyramid_decom(img=x)

        trans_pyrs = []

        for i in range(self.num_high + 1):
            trans_pyr = self.__getattr__('AE_{}'.format(i))(
                pyrs[-1 - i])
            trans_pyrs.append(trans_pyr)
        out = self.lap_pyramid.pyramid_recons(trans_pyrs)

        return out


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    x = torch.randn(1, 3, 640, 640).to(device)
    enhancement = DENet().to(device)
    x = enhancement(x)
    print(x.shape)

    x = torch.randn(1, 3, 640, 640).to(device)
    enhancement = PENet().to(device)
    x = enhancement(x)
    print(x.shape)
