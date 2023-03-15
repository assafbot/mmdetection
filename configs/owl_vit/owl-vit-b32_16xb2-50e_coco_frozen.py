_base_ = 'owl-vit-b32_16xb2-50e_coco_frozen.py'

model = dict(
    backbone=dict(
        frozen=True))