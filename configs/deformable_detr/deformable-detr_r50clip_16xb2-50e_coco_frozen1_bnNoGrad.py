_base_ = 'deformable-detr_r50clip_16xb2-50e_coco.py'


model = dict(
    backbone=dict(
        frozen_stages=1,
        bn_requires_grad=False
    )
)
