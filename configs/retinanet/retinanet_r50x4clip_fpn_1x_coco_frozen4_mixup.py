_base_ = './retinanet_r50x4clip_fpn_1x_coco_mixup.py'

# model settings
model = dict(
    backbone=dict(
        frozen_stages=4,
    )
)
