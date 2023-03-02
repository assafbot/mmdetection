_base_ = './retinanet_r50clip_fpn_1x_lvis-v1.py'

# model settings
model = dict(
    backbone=dict(
        model_name='RN50x4',
        init_cfg=dict(type='Pretrained', checkpoint='mentee://mmdetection/pretrained/RN50_openai.pth')
    )
)
