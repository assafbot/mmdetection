import torch
from mmengine.model import BaseModule, bias_init_with_prob
from torch import nn

from mmdet.models.layers.scale import ScaleLayer
from mmdet.registry import MODELS, DATASETS
from mmdet.utils.clip import canonicalize

try:
    import open_clip
except ImportError:
    open_clip = None


@MODELS.register_module()
class ConvClassPredictor(BaseModule):
    def __init__(self, in_channels, num_base_priors, out_channels,
                 kernel_size=1, padding=0,
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01, bias_prob=0.01)):
        super().__init__(init_cfg=init_cfg)
        self.conv = nn.Conv2d(in_channels, num_base_priors * out_channels,
                              kernel_size=kernel_size, padding=padding)

    def forward(self, tensor, query):
        if query is not None:
            raise NotImplementedError('need to support query')
        return self.conv(tensor)


@MODELS.register_module()
class LinearClassPredictor(BaseModule):
    def __init__(self, in_channels, out_channels, init_cfg=None, num_outputs=None):
        super().__init__(init_cfg=init_cfg)
        self.pred = nn.Linear(in_channels, num_outputs if num_outputs is not None else out_channels)

    def forward(self, tensor, query):
        logits = self.pred(tensor)
        if query is None:
            return logits

        not_valid = query == -1
        query = query.clone()
        query[not_valid] = 0
        idx = torch.arange(len(tensor), dtype=torch.int64)[:, None].repeat(1, query.shape[1])
        idx = idx.to(query)
        logits = logits[idx, :, query]
        logits[not_valid] = 0
        logits = logits.permute(0, 2, 1)
        return logits


class AbstractClipClassPredictor(BaseModule):
    def __init__(self, in_channels, out_channels,
                 model_name=None, pretrained=None, dataset_name='CocoDataset', class_names=None, canonicalize_text_labels=False, class_embeddings=None,
                 norm_text=True, norm_image=False, scale=True, bias=True, scale_with_input=False, templates=['{}'], reduction='avg',
                 **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.norm_image = norm_image
        self.norm_text = norm_text
        self.templates = templates
        self.reduction = reduction
        self.canonicalize_text_labels = canonicalize_text_labels

        self.class_embeddings = class_embeddings
        if self.class_embeddings is not None:
            from mmengine.runner import CheckpointLoader
            self.class_embeddings = CheckpointLoader.load_checkpoint(self.class_embeddings)

        if model_name is None or pretrained is None:
            raise ValueError(f'model_name and pretrained must be specified for {self.__class__.__name__}')

        self.model = [open_clip.create_model(model_name, pretrained=pretrained)]  # hack to remove from saved model
        self.model[0].eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.emb_dim = self.model[0].text_projection.shape[1]

        self.scale = ScaleLayer(scale=scale, bias=bias, use_input=scale_with_input, in_channels=in_channels)

        # TODO: @assaf support changing self.pred with a hook
        self.class_names = class_names or DATASETS.get(dataset_name).METAINFO['classes']
        self._set_pred(self.class_names)

    def to(self, device):
        super().to(device)
        self.model[0].to(device)

    def _set_pred(self, class_names):
        with torch.no_grad():
            class_embeddings = self._get_class_embeddings(class_names)
            num_classes = class_embeddings.shape[0] // len(self.templates)
            assert num_classes == self.out_channels, 'mismatch between expected output size and number of classes'
            pred = nn.Linear(self.emb_dim, class_embeddings.shape[0], bias=False)
            pred.weight[:] = class_embeddings
            pred.weight.requires_grad = False
            pred.to(next(self.parameters()).device)

        self._pred = pred

    def _get_class_embeddings(self, class_names):
        if self.canonicalize_text_labels:
            class_names = canonicalize(class_names)

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

    def forward(self, tensor, query):
        # if not hasattr(self, 'once') or not self.once:
        #     # class_names = DATASETS.get('CocoDataset').METAINFO['classes']
        #     # self.reduction = 'max'
        #     # self.templates = ['{}', 'a close-up photo of a {}.', 'a photo of the {}.', 'a photo of one {}.', 'a photo of a {}.',
        #     #     'a photo of a small {}.', 'a photo of a large {}.', 'a photo of the small {}.', 'a photo of the large {}.']
        #     # self.templates = ['{}', 'a photo of a {}.']
        #     mapping = {'grape': 'grapes'}
        #     self.class_names = [mapping.get(c, '') for c in self.class_names]
        #     # self.class_names = ['cake']  # * self.out_channels
        #     # self.class_names += [''] * (self.out_channels - len(self.class_names))
        #     print('Recomputing class names embeddings...', end='')
        #     self.class_embeddings = None
        #     self._set_pred(self.class_names)
        #     print(' Done!')
        #     self.once = True

        return self._forward(tensor, query)

    def pred(self, image_emb):
        if self.class_embeddings is not None:
            self.class_embeddings = self.class_embeddings.to(image_emb.device)
            if self.training:
                p = torch.randint(0, self.class_embeddings.shape[1], (self.class_embeddings.shape[2],))
                i = torch.arange(0, self.class_embeddings.shape[2])
                logits = image_emb @ self.class_embeddings[:, p, i]
            else:
                logits = image_emb @ self.class_embeddings[:, 0]
        else:
            logits = self._pred(image_emb)
        return logits

    def _forward(self, tensor, query):
        raise NotImplementedError()


@MODELS.register_module()
class ClipConvClassPredictor(AbstractClipClassPredictor):
    def __init__(self, in_channels, num_base_priors, out_channels,
                 init_cfg=[dict(type='Normal', layer='Conv2d', std=0.01),
                           dict(type='Constant', layer='ScaleLayer', val=1, bias=bias_init_with_prob(0.01))], **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, init_cfg=init_cfg, **kwargs)
        self.proj = nn.Conv2d(in_channels, num_base_priors * self.emb_dim, 3, padding=1)

    def _forward(self, x, query):
        if query is not None:
            raise NotImplementedError('need to support query')

        image_emb = self.proj(x)
        n, c, h, w = image_emb.shape
        image_emb = image_emb.permute(0, 2, 3, 1).reshape(n, h, w, -1, self.emb_dim)

        if self.norm_image:
            image_emb = image_emb / (image_emb.norm(p=2, dim=-1, keepdim=True) + 1e-6)

        logits = self.pred(image_emb)
        logits = self.scale(logits, x)
        logits = self.reduce(logits)

        logits = logits.reshape(n, h, w, -1).permute(0, 3, 1, 2)
        return logits


@MODELS.register_module()
class ClipLinearClassPredictor(AbstractClipClassPredictor):
    def __init__(self, in_channels, out_channels,
                 init_cfg=[dict(type='Constant', layer='ScaleLayer', val=1, bias=bias_init_with_prob(0.01))], **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, init_cfg=init_cfg, **kwargs)
        self.proj = nn.Linear(in_channels, self.emb_dim)

    def _forward(self, x, query):
        if query is not None:
            raise NotImplementedError('need to support query')

        image_emb = self.proj(x)
        if self.norm_image:
            image_emb = image_emb / (image_emb.norm(p=2, dim=-1, keepdim=True) + 1e-6)

        logits = self.pred(image_emb)
        logits = self.scale(logits, x)
        logits = self.reduce(logits)

        return logits


@MODELS.register_module()
class QueryClipLinearClassPredictor(BaseModule):
    def __init__(self, in_channels, out_channels,
                 model_name=None, pretrained=None, freeze_text=False, norm_text=True, norm_image=True, scale=True, bias=True,
                 ensemble_mode='avg',
                 init_cfg=[dict(type='Normal', layer='Conv2d', std=0.01),
                           dict(type='Constant', layer='ScaleLayer', val=10, bias=bias_init_with_prob(0.01))]):
        super().__init__(init_cfg=init_cfg)
        self.norm_image = norm_image
        self.norm_text = norm_text
        self.freeze_text = freeze_text
        self.ensemble_mode = ensemble_mode

        if model_name is None or pretrained is None:
            raise ValueError(f'model_name and pretrained must be specified for {self.__class__.__name__}')

        self.model = open_clip.create_model(model_name, pretrained=pretrained)
        self.model.visual = None
        self.model.logit_scale = None

        self.emb_dim = self.model.text_projection.shape[1]
        self.proj = nn.Linear(in_channels, self.emb_dim)
        self.scale = ScaleLayer(scale=scale, bias=bias, in_channels=in_channels)

        self._freeze()

    def train(self, mode=True):
        super().train(mode)
        self._freeze()

    def _freeze(self):
        if not self.freeze_text:
            return

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x, query):
        if query.shape[3] != self.emb_dim:
            text_emb = self.encode_query(query)
        else:
            text_emb = query

        image_emb = self.proj(x)
        if self.norm_image:
            image_emb = image_emb / (image_emb.norm(p=2, dim=-1, keepdim=True) + 1e-6)

        logits = torch.einsum('...wc,...qec->...weq', image_emb, text_emb)
        logits = self.scale(logits, x)

        n, w, e, q = logits.shape
        if e == 1:
            logits = logits.view(n, w, q)
        elif self.ensemble_mode == 'avg':
            logits = logits.mean(2)
        elif self.ensemble_mode == 'max':
            logits = logits.max(2)[0]
        else:
            raise ValueError(f'Unknowm ensemble mode {self.ensemble_mode}')

        return logits

    def encode_query(self, query):
        n, q, e, _ = query.shape
        text_emb = self.model.encode_text(query.flatten(0, 2))
        text_emb = text_emb.view(n, q, e, -1)
        if self.norm_text:
            text_emb = text_emb / (text_emb.norm(p=2, dim=-1, keepdim=True) + 1e-6)
        return text_emb
