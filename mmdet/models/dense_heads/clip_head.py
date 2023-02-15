import torch

from mmdet.datasets import CocoDataset
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.registry import MODELS

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    CLIPProcessor, CLIPModel = None, None


@MODELS.register_module()
class ClipHead(BaseDenseHead):
    def __new__(cls, bbox_head, **kwargs):
        if CLIPModel is None or CLIPProcessor is None:
            raise ImportError('Please run "pip install transformers" to use ClipHead')

        with torch.no_grad():
            # TODO: @assaf support any dataset / dynamic input
            class_names = CocoDataset.METAINFO['classes']
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            class_tokens = clip_processor(text=class_names, return_tensors="pt", padding=True)
            class_features = clip_model.text_model(**class_tokens)[1]
            class_embeddings = clip_model.text_projection(class_features)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True)

        num_classes, emb_dim = class_embeddings.shape

        bbox_head = bbox_head.copy()
        bbox_head.update(kwargs)
        assert bbox_head['num_classes'] == num_classes
        bbox_head['num_classes'] = emb_dim
        bbox_head = MODELS.build(bbox_head)
        # assert isinstance(bbox_head, DETRHead)

        def forward(*args, **kwargs):
            output = bbox_head._forward(*args, **kwargs)
            pred = torch.nn.functional.linear(output[0], class_embeddings.to(output[0].device))
            return (pred,) + output[1:]

        bbox_head._forward = bbox_head.forward
        bbox_head.forward = forward
        bbox_head.cls_out_channels = num_classes
        bbox_head.num_classes = num_classes

        return bbox_head
