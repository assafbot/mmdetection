_base_ = './retinanet_r50clip_fpn_1x_coco.py'

# model settings
model = dict(
    backbone=dict(
        frozen_stages=-1,
        norm_eval=False
    )
)
