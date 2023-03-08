import re
from random import choice

import torch
from mmengine.model import BaseModule, bias_init_with_prob
from torch import nn

from mmdet.models.layers.scale import ScaleLayer
from mmdet.registry import MODELS, DATASETS
from mmdet.utils.clip import TRAINING_PROMPT_TEMPLATES

try:
    import open_clip
except ImportError:
    open_clip = None


def _canonicalize(class_names):
    new_class_names = []
    for class_name in class_names:
        class_name = class_name.lower()
        class_name = re.sub(f'[^a-z0-9- ]', ' ', class_name)
        class_name = re.sub(r'\s+', ' ', class_name)
        class_name = re.sub(r'-+', '-', class_name)
        class_name = class_name.strip()
        new_class_names.append(class_name)

    return new_class_names

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
class LinearClassPredictor(BaseModule):
    def __init__(self, embed_dims, cls_out_channels, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.pred = nn.Linear(embed_dims, cls_out_channels)

    def forward(self, tensor):
        return self.pred(tensor)


def _add_random_template(class_name):
    return choice(TRAINING_PROMPT_TEMPLATES).format(class_name)


class AbstractClipClassPredictor(BaseModule):
    def __init__(self, cls_out_channels,
                 model_name=None, pretrained=None, dataset_name='CocoDataset', class_names=None, canonicalize_text_labels=False,
                 norm_text=True, norm_image=False, scale=True, bias=True, templates=['{}'], reduction='avg', random_template=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.cls_out_channels = cls_out_channels
        self.norm_image = norm_image
        self.norm_text = norm_text
        self.templates = templates
        self.reduction = reduction
        self.canonicalize_text_labels = canonicalize_text_labels
        self.random_template = random_template

        if model_name is None or pretrained is None:
            raise ValueError(f'model_name and pretrained must be specified for {self.__class__.__name__}')

        self.model = [open_clip.create_model(model_name, pretrained=pretrained)]  # hack to remove from saved model
        self.model[0].eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.emb_dim = self.model[0].text_projection.shape[1]

        self.scale = ScaleLayer(scale=scale, bias=bias)

        # TODO: @assaf support changing self.pred with a hook
        self.class_names = class_names or DATASETS.get(dataset_name).METAINFO['classes']
        self.pred = self._get_pred(self.class_names)

    def to(self, device):
        super().to(device)
        self.model[0].to(device)

    def _get_pred(self, class_names):
        with torch.no_grad():
            class_embeddings = self._get_class_embeddings(class_names)
            num_classes = class_embeddings.shape[0] // len(self.templates)
            assert num_classes == self.cls_out_channels, 'mismatch between expected output size and number of classes'
            pred = nn.Linear(self.emb_dim, class_embeddings.shape[0], bias=False)
            pred.weight[:] = class_embeddings
            pred.weight.requires_grad = False
            pred.to(next(self.parameters()).device)

        return pred

    def _get_class_embeddings(self, class_names):
        if self.canonicalize_text_labels:
            class_names = _canonicalize(class_names)

        with torch.no_grad():
            # TODO: @assaf support any dataset / dynamic input
            if open_clip is None:
                raise ImportError('Please run "pip install open_clip_torch" to use ClipHead')

            texts = [template.format(class_name) for template in self.templates for class_name in class_names]
            class_embeddings = self.model[0].encode_text(self.tokenizer(texts))
            if self.norm_text:
                class_embeddings = class_embeddings / class_embeddings.norm(p=2, dim=1, keepdim=True)
            class_embeddings = torch.nn.Parameter(class_embeddings, requires_grad=False)

        return class_embeddings

    def reduce(self, tensor):
        tensor = tensor.reshape(tensor.shape[:-1] + (len(self.templates), -1))
        if self.reduction == 'max':
            tensor, _ = tensor.max(-2)
        elif self.reduction == 'avg':
            tensor = tensor.mean(-2)
        else:
            raise NotImplementedError(f'Unknown reduction: {self.reduction}')
        return tensor

    def forward(self, tensor):
        # if not hasattr(self, 'once') or not self.once:
        #     # class_names = DATASETS.get('CocoDataset').METAINFO['classes']
        #     self.reduction = 'max'
        #     self.templates = ['{}', 'a close-up photo of a {}.', 'a photo of the {}.', 'a photo of one {}.', 'a photo of a {}.',
        #         'a photo of a small {}.', 'a photo of a large {}.', 'a photo of the small {}.', 'a photo of the large {}.']
        #     # self.templates = ['{}', 'a photo of a {}.']
        #     self.class_names = ['screen']  # * self.cls_out_channels
        #     self.class_names += [''] * (self.cls_out_channels - len(self.class_names))
        #     print('Recomputing class names embeddings...', end='')
        #     self.pred = self._get_pred(self.class_names)
        #     print(' Done!')
        #     self.once = True

        if self.random_template and self.training:
            assert self.templates == ['{}']
            class_names = [_add_random_template(c) for c in self.class_names]
            self.pred = self._get_pred(class_names)

        return self._forward(tensor)

    def _forward(self, tensor):
        raise NotImplementedError()



@MODELS.register_module()
class ClipConvClassPredictor(AbstractClipClassPredictor):
    def __init__(self, in_channels, num_base_priors, cls_out_channels,
                 init_cfg=[dict(type='Normal', layer='Conv2d', std=0.01),
                           dict(type='Constant', layer='ScaleLayer', val=1, bias=bias_init_with_prob(0.01))], **kwargs):
        super().__init__(cls_out_channels=cls_out_channels, init_cfg=init_cfg, **kwargs)
        self.proj = nn.Conv2d(in_channels, num_base_priors * self.emb_dim, 3, padding=1)

    def _forward(self, tensor):
        tensor = self.proj(tensor)
        n, c, h, w = tensor.shape
        tensor = tensor.permute(0, 2, 3, 1).reshape(n, h, w, -1, self.emb_dim)

        if self.norm_image:
            tensor = tensor / tensor.norm(p=2, dim=-1, keepdim=True)

        tensor = self.pred(tensor)
        tensor = self.scale(tensor)
        tensor = self.reduce(tensor)

        tensor = tensor.reshape(n, h, w, -1).permute(0, 3, 1, 2)
        return tensor


@MODELS.register_module()
class ClipLinearClassPredictor(AbstractClipClassPredictor):
    def __init__(self, embed_dims, cls_out_channels,
                 init_cfg=[dict(type='Constant', layer='ScaleLayer', val=1, bias=bias_init_with_prob(0.01))], **kwargs):
        super().__init__(cls_out_channels=cls_out_channels, init_cfg=init_cfg, **kwargs)
        self.proj = nn.Linear(embed_dims, self.emb_dim)

    def _forward(self, tensor):
        tensor = self.proj(tensor)
        if self.norm_image:
            tensor = tensor / tensor.norm(p=2, dim=-1, keepdim=True)

        tensor = self.pred(tensor)
        tensor = self.scale(tensor)
        tensor = self.reduce(tensor)

        return tensor
