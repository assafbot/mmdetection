import torch
from mmengine.model import BaseModule
from torch import nn

from mmdet.datasets import CocoDataset
from mmdet.registry import MODELS

try:
    import open_clip
except ImportError:
    open_clip = None


@MODELS.register_module()
class ConvClassPredictor(BaseModule):
    def __init__(self, in_channels, num_base_priors, cls_out_channels,
                 kernel_size=1, padding=0,
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01, bias_prob=0.01)):
        super().__init__(init_cfg=init_cfg)
        self.conv = nn.Conv2d(in_channels, num_base_priors * cls_out_channels,
                              kernel_size=kernel_size, padding=padding)

    def forward(self, tensor):
        return self.conv(tensor)
