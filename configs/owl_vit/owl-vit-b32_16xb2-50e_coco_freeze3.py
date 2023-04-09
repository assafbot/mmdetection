_base_ = 'owl-vit-b32_16xb2-50e_coco.py'

model = dict(
    backbone=dict(
        frozen_stages=3))