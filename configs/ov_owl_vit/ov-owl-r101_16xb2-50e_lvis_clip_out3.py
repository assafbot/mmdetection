_base_ = 'ov-owl-r101_16xb2-50e_lvis_clip.py'

model = dict(
    backbone=dict(
        out_indices=(3,)))
