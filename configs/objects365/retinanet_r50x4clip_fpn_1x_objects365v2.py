_base_ = './retinanet_r50clip_fpn_1x_objects365v2.py'

# model settings
model = dict(
    backbone=dict(
        model_name='RN50x4',
        init_cfg=dict(type='Pretrained', checkpoint='mentee://mmdetection/pretrained/RN50x4_openai.pth')
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
