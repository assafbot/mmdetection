_base_ = 'ov-owl-vit-b32_16xb2-50e_lvis.py'

model = dict(backbone=dict(pretrained=None))

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys=dict({
            'backbone': dict(lr_mult=1)
        })))