_base_ = 'deformable-detr_r50_16xb2-50e_coco_clip.py'


model = dict(
    type='DeformableDETR',
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
    )
)
