_base_ = 'deformable-detr_r50_16xb2-50e_lvis_federated_clip_v2.py'

model = dict(
    as_two_stage=False,
    with_box_refine=True,
    neck=dict(extra_convs_kernel=1),
    encoder=dict(num_layers=1),
    decoder=dict(num_layers=1),
    backbone=dict(pretrained='openai'),
    bbox_head=dict(fc_cls=dict(class_embeddings='mentee://mmdetection/embeddings/LVISV1Dataset_classes1203_RN50_openai_normalize_embeddings_templates81.pth'))
)
