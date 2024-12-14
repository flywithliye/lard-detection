# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""

from .block import (C1, C2, C3, C3TR, DFL, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, GhostBottleneck,
                    HGBlock, HGStem, Proto, RepC3)
from .conv import (CBAM, ChannelAttention, Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d, Focus,
                   GhostConv, LightConv, RepConv, SpatialAttention)
from .head import Classify, Detect, Pose, RTDETRDecoder, Segment
from .attention import (SE, ECA, eSE, GAM, SA, CPCA, EMA, TA, LSK, LSKA, C2f_SE, C2f_CBAM, C2f_ECA, C2f_eSE, C2f_GAM, 
                        C2f_SA, C2f_CPCA, C2f_EMA, C2f_TA, C2f_LSK, C2f_LSKA)
from .neck import BiFPN_Add2, BiFPN_Add3, BiFPN_Concat2, BiFPN_Concat3, ASFF2, ASFF3, ASFF4
from .upsample import CARAFE, DySample
from .enhence import DENet, PENet
from .transformer import (AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, LayerNorm2d,
                          MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer, TransformerLayer)

__all__ = ('Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus',
           'GhostConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'TransformerLayer',
           'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
           'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect',
           'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'AIFI',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP', 
           'SE', 'ECA', 'eSE', 'GAM', 'SA', 'CPCA', 'EMA', 'TA', 'LSK', 'LSKA', 
           'C2f_SE', 'C2f_CBAM', 'C2f_ECA', 'C2f_eSE', 'C2f_GAM', 
           'C2f_SA', 'C2f_CPCA', 'C2f_EMA', 'C2f_TA', 'C2f_LSK', 'C2f_LSKA'
           'BiFPN_Add2', 'BiFPN_Add3', 'BiFPN_Concat2', 'BiFPN_Concat3', 'ASFF2', 'ASFF3', 'ASFF4',
           'CARAFE', 'DySample',
           'DENet', 'PENet')
