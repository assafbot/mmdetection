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
class ClipViT(BaseModule):
    def __init__(self, model_name, pretrained=None, init_cfg=None, frozen=False):
        super(ClipViT, self).__init__(init_cfg)
        if open_clip is None:
            raise ImportError(f'Please run "pip install open_clip_torch" to use {self.__class__.__name__}')

        self.model = open_clip.create_model(model_name, pretrained)
        self.model.visual.output_tokens = True

        # Save original parameters
        self.positional_embedding = [self.model.visual.positional_embedding]  # Hack to hide parameter
        self.grid_size = self.model.visual.grid_size
        self.image_size = self.model.visual.image_size

        self.frozen = frozen
        self._freeze()

    def _freeze(self):
        self.eval()

    def forward(self, x):
        vit = self.model.visual

        image_size = x.shape[2:]
        image_height, image_width = vit.image_size = image_size
        patch_height, patch_width = vit.patch_size
        vit.grid_size = (image_height // patch_height, image_width // patch_width)

        with torch.no_grad():
            pe = self.positional_embedding[0]
            assert pe.shape[0] == self.grid_size[0] * self.grid_size[1] + 1
            cls_emb = pe[:1]
            grid_emb = pe[1:]
            grid_emb = grid_emb.reshape(*self.grid_size[::-1], -1)
            grid_emb = grid_emb.permute(2, 0, 1)[None]
            grid_emb = torch.nn.functional.interpolate(grid_emb, vit.grid_size, mode='bilinear')
            grid_emb = grid_emb[0].permute(1, 2, 0)
            grid_emb = grid_emb.flatten(0, 1)
            pe = torch.cat((cls_emb, grid_emb))
            vit.positional_embedding = torch.nn.Parameter(pe, requires_grad=False)

        _, features = vit(x)
        features = features @ vit.proj
        n, _, c = features.shape
        features = features.reshape(n, *self.model.visual.grid_size, c)
        features = features.permute(0, 3, 1, 2)
        return [features]

    def train(self, mode=True):
        if self.frozen:
            mode = False

        super(ClipViT, self).train(mode)

        if self.frozen:
            for param in self.parameters():
                param.requires_grad = False


@MODELS.register_module()
class ClipResNet(BaseModule):
    def __init__(self, model_name, pretrained=None, out_indices=None, frozen_stages=-1, norm_eval=True, init_cfg=None):
        super(ClipResNet, self).__init__(init_cfg)
        if open_clip is None:
            raise ImportError(f'Please run "pip install open_clip_torch" to use {self.__class__.__name__}')

        model = open_clip.create_model(model_name, pretrained)
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
