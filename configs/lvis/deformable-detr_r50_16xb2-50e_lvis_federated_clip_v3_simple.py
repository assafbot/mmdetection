_base_ = 'deformable-detr_r50_16xb2-50e_lvis_federated_clip_v3.py'

model = dict(
    as_two_stage=False,
    with_box_refine=False
)
