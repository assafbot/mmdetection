import torch
from mmengine.model import BaseModule, bias_init_with_prob
from torch import nn

from mmdet.models.layers.scale import ScaleLayer
from mmdet.registry import MODELS, DATASETS

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
                 model_name=None, pretrained=None, dataset_name='CocoDataset', class_names=None,
                 norm_text=True, norm_image=False, scale=True, bias=True,
                 init_cfg=[dict(type='Normal', layer='Conv2d', std=0.01),
                           dict(type='Constant', layer='ScaleLayer', val=1, bias=bias_init_with_prob(0.01))]):
        super().__init__(init_cfg=init_cfg)
        self.norm_image = norm_image
        self.norm_text = norm_text
        self.cls_out_channels = cls_out_channels

        if model_name is None or pretrained is None:
            raise ValueError(f'model_name and pretrained must be specified for {self.__class__.__name__}')

        self.model = [open_clip.create_model(model_name, pretrained=pretrained)]  # hack to remove from saved model
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.emb_dim = self.model[0].text_projection.shape[1]

        self.proj = nn.Conv2d(in_channels, num_base_priors * self.emb_dim, 3, padding=1)
        self.scale = ScaleLayer(scale=scale, bias=bias)

        # TODO: @assaf support changing self.pred with a hook
        self.class_names = class_names or DATASETS.get(dataset_name).METAINFO['classes']
        self.pred = self._get_pred(self.class_names)

    def _get_pred(self, class_names):
        class_embeddings = self._get_class_embeddings(class_names)
        num_classes = class_embeddings.shape[0]
        assert num_classes == self.cls_out_channels, 'mismatch between expected output size and number of classes'
        pred = nn.Linear(self.emb_dim, num_classes, bias=False)
        with torch.no_grad():
            pred.weight[:] = class_embeddings
        pred.weight.requires_grad = False
        return pred

    def _get_class_embeddings(self, class_names):
        with torch.no_grad():
            # TODO: @assaf support any dataset / dynamic input
            if open_clip is None:
                raise ImportError('Please run "pip install open_clip_torch" to use ClipHead')

            class_embeddings = self.model[0].encode_text(self.tokenizer(class_names))
            if self.norm_text:
                class_embeddings = class_embeddings / class_embeddings.norm(p=2, dim=1, keepdim=True)
            class_embeddings = torch.nn.Parameter(class_embeddings, requires_grad=False)

        return class_embeddings

    def forward(self, tensor):
        # if not hasattr(self, 'once'):
        #     # class_names = DATASETS.get('CocoDataset').METAINFO['classes']
        #     class_names = ['pillow'] + [''] * 79
        #     self.pred = self._get_pred(class_names)
        #     self.once = True

        tensor = self.proj(tensor)
        n, c, h, w = tensor.shape
        tensor = tensor.permute(0, 2, 3, 1).reshape(n, h, w, -1, self.emb_dim)

        if self.norm_image:
            tensor = tensor / tensor.norm(p=2, dim=-1, keepdim=True)

        tensor = self.pred(tensor)
        tensor = self.scale(tensor)

        tensor = tensor.reshape(n, h, w, -1).permute(0, 3, 1, 2)
        return tensor
