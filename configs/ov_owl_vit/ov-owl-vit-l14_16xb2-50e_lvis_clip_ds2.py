_base_ = 'ov-owl-vit-l14_16xb2-50e_lvis_clip_ds2.py'

model_name = 'ViT-L-14'
pretrained = 'openai'

model = dict(
    backbone=dict(
        model_name=model_name,
        pretrained=pretrained),
    bbox_head=dict(
        fc_cls=dict(
            model_name=model_name,
            pretrained=pretrained)))
