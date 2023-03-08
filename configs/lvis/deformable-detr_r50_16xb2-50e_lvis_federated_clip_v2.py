_base_ = 'deformable-detr_r50_16xb2-50e_lvis_federated_clip.py'

train_pipeline = _base_.train_dataloader.dataset.pipeline
train_pipeline = train_pipeline[:-1] + [
    dict(type='RemoveLVISRareLabels'),
    dict(type='AddRandomNegatives', num_classes=_base_.model.bbox_head.num_classes, total=50),
] + train_pipeline[-1:]

model = dict(
    bbox_head=dict(
        fc_cls=dict(
            canonicalize_text_labels=True)))

train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline
    )
)