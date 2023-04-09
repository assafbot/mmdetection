_base_ = './retinanet_r50clip_fpn_1x_coco.py'

# model settings
model = dict(
    backbone=dict(
        norm_eval=False,
    )
)
