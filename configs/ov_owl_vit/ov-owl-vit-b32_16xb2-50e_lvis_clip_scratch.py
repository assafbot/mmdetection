_base_ = 'ov-owl-vit-b32_16xb2-50e_lvis_clip.py'

model = dict(backbone=dict(pretrained=None))

optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys=dict({
            'backbone': dict(lr_mult=1),
            'bbox_head.fc_cls.model': dict(lr_mult=0.01)
        })))