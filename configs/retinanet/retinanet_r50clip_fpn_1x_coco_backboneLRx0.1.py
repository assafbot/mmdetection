_base_ = './retinanet_r50clip_fpn_1x_coco.py'

# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1)
        }))
