_base_ = 'ov-owl-vit-b32_16xb2-50e_coco.py'

model_name = 'ViT-B-16'
pretrained = 'openai'

model = dict(
    backbone=dict(
        model_name=model_name,
        pretrained=pretrained))