_base_ = './retinanet_r50clip_fpn_1x_coco_frozen.py'

# model settings
model = dict(
    backbone=dict(
        frozen_stages=-1,
        norm_eval=False,
    )
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)

