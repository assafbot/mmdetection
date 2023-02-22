_base_ = './retinanet_r50_fpn_1x_coco_clip.py'

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
        frozen_stages=1,
        norm_eval=True
    ),
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[256, 640, 1280, 2560],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5)
)