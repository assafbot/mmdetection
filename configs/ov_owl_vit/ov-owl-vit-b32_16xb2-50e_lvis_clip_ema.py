_base_ = 'ov-owl-vit-b32_16xb2-50e_lvis_clip.py'

_base_.custom_hooks += [
    dict(type='EMAHookV2',
         ema_type='ExponentialMovingAverageV2',
         momentum=0.0002,
         update_buffers=True,
         priority=49)]
