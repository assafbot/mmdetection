_base_ = 'ov-owl-vit-b32_16xb2-50e_lvis_clip.py'

pretrained = None

model = dict(
    backbone=dict(
        pretrained=pretrained),
    bbox_head=dict(
        fc_cls=dict(
            pretrained=pretrained)))
