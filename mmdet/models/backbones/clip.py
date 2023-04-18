# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmengine.model import BaseModule, ModuleList
from torch.nn.modules.batchnorm import _BatchNorm, _NormBase

from mmdet.registry import MODELS

try:
    import open_clip
except ImportError:
    open_clip = None


@MODELS.register_module()
class ClipViT(BaseModule):
    def __init__(self, model_name, pretrained=None, init_cfg=None, frozen_stages=-1, norm_eval=False,
                 final_ln_post=True, pos_emb_requires_grad=False):
        super(ClipViT, self).__init__(init_cfg)
        if open_clip is None:
            raise ImportError(f'Please run "pip install open_clip_torch" to use {self.__class__.__name__}')

        model = open_clip.create_model(model_name, pretrained)
        self.vit = model.visual
        self.vit.output_tokens = True

        # self.proj = self.vit.proj  # TODO: @assaf do I need to use proj?
        self.vit.proj = None

        self.ln_post = self.vit.ln_post
        self.final_ln_post = final_ln_post  # this might be a bug, kept only for backward compatibility
        self.vit.ln_post = torch.nn.Identity()

        # Save original parameters
        self.positional_embedding = torch.nn.Parameter(self.vit.positional_embedding, requires_grad=pos_emb_requires_grad)
        delattr(self.vit, 'positional_embedding')
        self.grid_size = self.vit.grid_size
        self.image_size = self.vit.image_size

        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self._freeze_stages()

    def forward(self, x):
        image_size = x.shape[2:]
        image_height, image_width = self.vit.image_size = image_size
        patch_height, patch_width = self.vit.patch_size
        self.vit.grid_size = (image_height // patch_height, image_width // patch_width)

        with torch.set_grad_enabled(self.positional_embedding.requires_grad):
            pe = self.positional_embedding
            assert pe.shape[0] == self.grid_size[0] * self.grid_size[1] + 1
            cls_emb = pe[:1]
            grid_emb = pe[1:]
            grid_emb = grid_emb.reshape(*self.grid_size[::-1], -1)
            grid_emb = grid_emb.permute(2, 0, 1)[None]
            grid_emb = torch.nn.functional.interpolate(grid_emb, self.vit.grid_size, mode='bilinear')
            grid_emb = grid_emb[0].permute(1, 2, 0)
            grid_emb = grid_emb.flatten(0, 1)
            pe = torch.cat((cls_emb, grid_emb))
            self.vit.positional_embedding = pe

        token, features = self.vit(x)

        combined = torch.cat((token[:, None], features), dim=1)
        combined = self.ln_post(combined)

        token, features = combined[:, :1], combined[:, 1:]
        features = features * token
        if self.final_ln_post:  # this might be a bug to reuse the same LN layer
            features = self.ln_post(features)

        n, _, c = features.shape
        features_map = features.reshape(n, *self.vit.grid_size, c).permute(0, 3, 1, 2)
        return tuple([features_map])

    def _freeze_stages(self):
        vit = self.vit
        if self.frozen_stages >= 0:
            vit.positional_embedding.requires_grad = False
            vit.class_embedding.requires_grad = False
            for m in [vit.patchnorm_pre_ln, vit.conv1, vit.patch_dropout, vit.ln_pre]:
                self._freeze_layer(m)

        for i in range(1, self.frozen_stages + 1):
            m = vit.transformer.resblocks[i - 1]
            self._freeze_layer(m)

        if self.frozen_stages == len(vit.transformer.resblocks):
            if vit.proj:
                vit.proj.requires_grad = False

            for m in [vit.attn_pool, vit.ln_post, vit.ln_post, self.ln_post]:
                if m is None:
                    continue
                self._freeze_layer(m)

    def _freeze_layer(self, m):
        m.eval()
        for param in m.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        super(ClipViT, self).train(mode)
        self._freeze_stages()

        # trick: eval have effect on NormBase only
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _NormBase):
                    self._freeze_layer(m)


@MODELS.register_module()
class ClipResNet(BaseModule):
    def __init__(self, model_name, pretrained=None, out_indices=None, frozen_stages=-1, norm_eval=True,
                 bn_requires_grad=True, use_attn_pool=False, init_cfg=None):
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
        self.attnpool = rn.attnpool if use_attn_pool else None
        self.attnpool_ln = torch.nn.LayerNorm(self.attnpool.c_proj.out_features) if use_attn_pool else None
        self.out_indices = list(sorted(out_indices or [len(self.layers)-1 if not use_attn_pool else len(self.layers)]))
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        if not bn_requires_grad:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    for param in m.parameters():
                        param.requires_grad = False

        self._freeze_stages()

    def _freeze_stages(self):
        layers = self.layers + [self.attnpool]
        for i in range(max(self.frozen_stages, -1) + 1):
            m = layers[i]
            self._freeze_layer(m)

    def _freeze_layer(self, m):
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
        max_idx = max(self.out_indices)
        outputs = []
        for idx, layer in enumerate(self.layers[:max_idx+1]):
            x = layer(x)
            if idx in self.out_indices:
                outputs.append(x)

        if self.attnpool and len(self.layers) in self.out_indices:
            ap = self.attnpool
            n, c, h, w = x.shape
            pe = ap.positional_embedding

            cls_emb = pe[:1]
            grid_emb = pe[1:]
            grid_emb = grid_emb.reshape(7, 7, -1)
            grid_emb = grid_emb.permute(2, 0, 1)[None]
            grid_emb = torch.nn.functional.interpolate(grid_emb, (h, w), mode='bilinear')
            grid_emb = grid_emb[0].permute(1, 2, 0)
            grid_emb = grid_emb.flatten(0, 1)
            pe = torch.cat((cls_emb, grid_emb))

            x = x.reshape(n, c, h * w).permute(2, 0, 1)  # NCHW -> (HW)NC
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
            x = x + pe[:, None, :].to(x.dtype)  # (HW+1)NC
            x, _ = torch.nn.functional.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=c,
                num_heads=ap.num_heads,
                q_proj_weight=ap.q_proj.weight,
                k_proj_weight=ap.k_proj.weight,
                v_proj_weight=ap.v_proj.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat([ap.q_proj.bias, ap.k_proj.bias, ap.v_proj.bias]),
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=0.,
                out_proj_weight=ap.c_proj.weight,
                out_proj_bias=ap.c_proj.bias,
                use_separate_proj_weight=True,
                training=ap.training,
                need_weights=False
            )

            x = self.attnpool_ln(x)
            features = x[1:]
            cls = x[:1]
            features = features * cls
            features = features.reshape(h, w, n, -1).permute(2, 3, 0, 1)
            outputs.append(features)

        return tuple(outputs)
