_base_ = [
    '../_base_/datasets/lvis_v1_federated.py',
    '../_base_/models/deformable-detr_r50_refine_twostages.py',
    '../_base_/default_runtime.py'
]
model = dict(
    bbox_head=dict(
        num_classes=1203,
        use_category_ids=True))
