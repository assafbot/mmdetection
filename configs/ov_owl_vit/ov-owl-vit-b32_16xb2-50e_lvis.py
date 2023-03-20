_base_ = [
    '../_base_/datasets/lvis_v1_ov.py',
    '../_base_/models/owl-vit_b32-ov.py',
    '../_base_/schedules/schedule_50e.py',
    '../_base_/default_runtime.py'
]

model = dict(
    bbox_head=dict(
        num_classes=_base_.num_queries,
        fc_cls=dict(
            type='LinearClassPredictor',
            num_outputs=1203
        )
    )
)