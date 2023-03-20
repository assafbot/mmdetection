_base_ = 'ov-owl-vit-b32_16xb2-50e_lvis_clip.py'

optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys=dict({
            'backbone': dict(lr_mult=0.1),
            'bbox_head.fc_cls.model': dict(lr_mult=0.01)
        })))