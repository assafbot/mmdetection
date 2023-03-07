_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/lvis_v1_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    '../retinanet/retinanet_tta.py'
]

model = dict(
    bbox_head=dict(num_classes=1203, use_category_ids=True),
    test_cfg=dict(
        score_thr=0.01,
        max_per_img=300))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))
