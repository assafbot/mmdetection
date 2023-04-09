_base_ = 'ov-owl-vit-b32_16xb2-50e_lvis.py'

model = dict(
    bbox_head=dict(
        ignore_non_exhaustive_falses=True))