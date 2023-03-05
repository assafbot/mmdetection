_base_ = './retinanet_r50_fpn_1x_objects365v2.py'

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
        out_indices=(1, 2, 3, 4),
        model_name='RN50',
        frozen_stages=4,
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='mentee://mmdetection/pretrained/RN50_openai.pth')
    )
)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))
