_base_ = 'ov-owl-vit-b32_16xb2-50e_lvis_clip_ds2.py'

model_name = 'ViT-L-14'
pretrained = 'openai'

model = dict(
    backbone=dict(
        model_name=model_name,
        pretrained=pretrained),
    bbox_head=dict(
        embed_dims=1024,
        fc_cls=dict(
            model_name=model_name,
            pretrained=pretrained)))
