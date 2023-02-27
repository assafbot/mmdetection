# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmengine.model import BaseModule, ModuleList
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.registry import MODELS

try:
    import open_clip
except ImportError:
    open_clip = None


@MODELS.register_module()
class ClipResNet(BaseModule):
    def __init__(self, init_cfg=None, out_indices=None, frozen_stages=-1, norm_eval=True):
        super(ClipResNet, self).__init__(init_cfg)
        if open_clip is None:
            raise ImportError(f'Please run "pip install open_clip_torch" to use {self.__class__.__name__}')

        model, _, _ = open_clip.create_model_and_transforms('RN50x4', pretrained='openai')
        rn = model.visual
        stem = torch.nn.Sequential(rn.conv1, rn.bn1, rn.act1,
                                   rn.conv2, rn.bn2, rn.act2,
                                   rn.conv3, rn.bn3, rn.act3,
                                   rn.avgpool)
        self.layers = ModuleList([stem, rn.layer1, rn.layer2, rn.layer3, rn.layer4])
        self.out_indices = list(sorted(out_indices or [len(self.layers)-1]))
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self._freeze_stages()

    def _freeze_stages(self):
        for i in range(max(self.frozen_stages, -1) + 1):
            m = self.layers[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(ClipResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        outputs = []
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx in self.out_indices:
                outputs.append(x)

        return tuple(outputs)
