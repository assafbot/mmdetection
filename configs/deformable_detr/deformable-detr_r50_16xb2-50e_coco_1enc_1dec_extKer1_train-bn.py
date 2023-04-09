_base_ = 'deformable-detr_r50_16xb2-50e_coco_1enc_1dec_extKer1.py'

model = dict(
    backbone=dict(
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False))