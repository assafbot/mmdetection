import torch
from mmengine.model import BaseModule, bias_init_with_prob
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


@MODELS.register_module()
class ClipConvClassPredictor(BaseModule):
    def __init__(self, in_channels, num_base_priors, cls_out_channels,
                 norm_text=True, norm_image=True, scale_and_bias=False,
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01, bias_prob=0.01)):
        super().__init__(init_cfg=init_cfg)

        with torch.no_grad():
            # TODO: @assaf support any dataset / dynamic input
            class_names = CocoDataset.METAINFO['classes']

            if open_clip is None:
                raise ImportError('Please run "pip install open_clip_torch" to use ClipHead')
            model, _, _ = open_clip.create_model_and_transforms('RN50x4', pretrained='openai')
            tokenizer = open_clip.get_tokenizer('RN50x4')
            class_embeddings = model.encode_text(tokenizer(class_names))
            if norm_text:
                class_embeddings = class_embeddings / class_embeddings.norm(p=2, dim=1, keepdim=True)
            class_embeddings = torch.nn.Parameter(class_embeddings, requires_grad=False)

        num_classes, self.emb_dim = class_embeddings.shape
        assert num_classes == cls_out_channels
        self.proj = nn.Conv2d(in_channels, num_base_priors * self.emb_dim, 3, padding=1)
        self.pred = nn.Linear(self.emb_dim, num_classes, bias=False)
        with torch.no_grad():
            self.pred.weight[:] = class_embeddings
        self.pred.weight.requires_grad = False
        self.norm_image = norm_image

        if scale_and_bias:
            with torch.no_grad():
                self.scale = nn.Parameter(10 * torch.ones((1, )))
                self.bias = nn.Parameter(bias_init_with_prob(0.01) * torch.ones((1, )))
        else:
            self.scale = None
            self.bias = None

    def forward(self, tensor):
        tensor = self.proj(tensor)
        n, c, h, w = tensor.shape
        tensor = tensor.permute(0, 2, 3, 1).reshape(n, h, w, -1, self.emb_dim)

        if self.norm_image:
            tensor = tensor / tensor.norm(p=2, dim=-1, keepdim=True)
        tensor = self.pred(tensor)
        if self.scale is not None:
            tensor = tensor * self.scale
        if self.bias is not None:
            tensor = tensor + self.bias

        tensor = tensor.reshape(n, h, w, -1).permute(0, 3, 1, 2)
        return tensor
