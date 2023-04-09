_base_ = './retinanet_r50clip_fpn_1x_lvis-v1.py'

# model settings
model = dict(
    backbone=dict(
        frozen_stages=4,
    )
)
