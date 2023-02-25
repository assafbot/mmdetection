import torch

from mmdet.datasets import CocoDataset
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.dense_heads.retina_head import RetinaHead
from mmdet.registry import MODELS

try:
    import open_clip
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    open_clip = None
    CLIPProcessor, CLIPModel = None, None


@MODELS.register_module()
class ClipHead(BaseDenseHead):
    def __new__(cls, bbox_head, **kwargs):
        with torch.no_grad():
            # TODO: @assaf support any dataset / dynamic input
            class_names = CocoDataset.METAINFO['classes']

            if True:
                if open_clip is None:
                    raise ImportError('Please run "pip install open_clip_torch" to use ClipHead')
                model, _, _ = open_clip.create_model_and_transforms('RN50x4', pretrained='openai')
                tokenizer = open_clip.get_tokenizer('RN50x4')
                class_embeddings = model.encode_text(tokenizer(class_names))
            else:
                if CLIPModel is None or CLIPProcessor is None:
                    raise ImportError('Please run "pip install transformers" to use ClipHead')
                clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                class_tokens = clip_processor(text=class_names, return_tensors="pt", padding=True)
                class_features = clip_model.text_model(**class_tokens)[1]
                class_embeddings = clip_model.text_projection(class_features)

            class_embeddings = class_embeddings / class_embeddings.norm(p=2, dim=1, keepdim=True)
            class_embeddings = torch.nn.Parameter(class_embeddings, requires_grad=False)

        num_classes, emb_dim = class_embeddings.shape

        bbox_head = bbox_head.copy()
        bbox_head.update(kwargs)
        assert bbox_head['num_classes'] == num_classes
        bbox_head['num_classes'] = emb_dim
        bbox_head = MODELS.build(bbox_head)
        bbox_head.register_parameter('clip_class_embeddings', class_embeddings)
        assert isinstance(bbox_head, RetinaHead)  # TODO: @assaf could this simply be AnchorHead?
        assert bbox_head.use_sigmoid_cls

        def forward_retinanet(*args, **kwargs):
            output = bbox_head._forward(*args, **kwargs)
            for idx, pred in enumerate(output[0]):
                n, c, h, w = pred.shape
                pred = pred.permute(0, 2, 3, 1).reshape(n, h, w, -1, emb_dim)
                pred = pred / pred.norm(p=2, dim=-1, keepdim=True)
                pred = torch.nn.functional.linear(pred, bbox_head.clip_class_embeddings)
                pred *= 100
                pred = pred.reshape(n, h, w, -1).permute(0, 3, 1, 2)
                output[0][idx] = pred

            return output

        # def forward_dino(*args, **kwargs):
        #     output = bbox_head._forward(*args, **kwargs)
        #     pred = torch.nn.functional.linear(output[0], class_embeddings.to(output[0].device))
        #     return (pred,) + output[1:]

        bbox_head._forward = bbox_head.forward
        bbox_head.forward = forward_retinanet
        bbox_head.cls_out_channels = num_classes
        bbox_head.num_classes = num_classes

        return bbox_head
