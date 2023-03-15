_base_ = 'deformable-detr_r50clip_16xb2-50e_coco_frozen1_bnNoGrad.py'


model = dict(
    backbone=dict(
        use_attn_pool=True,
        out_indices=(2, 3, 5),
    ),
    neck=dict(
        in_channels=[512, 1024, 1024],
    )
)
