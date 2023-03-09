_base_ = 'deformable-detr_r50_16xb2-50e_lvis_federated_clip_v2.py'

train_pipeline = _base_.train_dataloader.dataset.pipeline
for p in train_pipeline:
    if p['type'] == 'AddRandomNegatives':
        p['type'] = 'AddRandomNegativesV2'

train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline
    )
)

model = dict(
    bbox_head=dict(
        fc_cls=dict(class_embeddings='mentee://mmdetection/pretrained/LVIS1023_embeddings_RN50_with-clip-templates.pth',
                    scale_with_input=True,
                    init_cfg=[dict(type='Normal', layer='Conv2d', std=1e-6)])
    )
)
