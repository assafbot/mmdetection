_base_ = 'owl-vit_b32.py'

model = dict(
    backbone=dict(
        type='ClipResNet',
        model_name='RN50',
        pretrained='openai',
        use_attn_pool=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    bbox_head=dict(
        embed_dims=1024,
    ))
