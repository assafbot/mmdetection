_base_ = [
    '../_base_/datasets/lvis_v1_unbalanced_no-rare.py',
    '../_base_/models/deformable-detr_r50_refine_twostages.py',
    '../_base_/default_runtime.py'
]
model = dict(
    bbox_head=dict(
        num_classes=1203,
        as_two_stage=False))
