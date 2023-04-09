_base_ = [
    '../_base_/datasets/lvis_v1_unbalanced.py',
    '../_base_/models/owl-vit_b32.py',
    '../_base_/schedules/schedule_50e.py',
    '../_base_/default_runtime.py'
]

model = dict(
    bbox_head=dict(
        num_classes=1203),
    test_cfg=dict(
        max_per_img=300))  # TODO: test with larger values of max_per_image?
