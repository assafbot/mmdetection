_base_ = 'owl-vit_b32.py'

model = dict(
    backbone=dict(
        _delete_=True,
        type='ClipResNet',
        model_name='RN50',
        pretrained='openai',
        frozen_stages=1,
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint='mentee://mmdetection/pretrained/RN50_openai.pth')),
    bbox_head=dict(
        embed_dims=2048,
    ))
