from mmengine.model import bias_init_with_prob

_base_ = 'deformable-detr_r50_16xb2-50e_lvis_federated.py'


# model settings
model = dict(
    data_preprocessor=dict(
        _delete_=True,
        type='DetDataPreprocessor',
        mean=[122.7709383, 116.7460125, 104.09373615],
        std=[68.5005327, 66.6321579, 70.32316305],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        _delete_=True,
        type='ClipResNet',
        out_indices=(2, 3, 4),
        model_name='RN50',
        frozen_stages=4,
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='mentee://mmdetection/pretrained/RN50_openai.pth')
    ),
    bbox_head=dict(
        fc_cls=dict(
            type='ClipLinearClassPredictor',
            norm_image=True,
            model_name='RN50',
            pretrained='openai',
            dataset_name='LVISV1Dataset',
            init_cfg=[dict(type='Normal', layer='Conv2d', std=0.01),
                      dict(type='Constant', layer='ScaleLayer', val=10, bias=bias_init_with_prob(0.01))]
        )
    )
)
