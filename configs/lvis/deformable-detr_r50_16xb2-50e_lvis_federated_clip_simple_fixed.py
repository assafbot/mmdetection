_base_ = 'deformable-detr_r50_16xb2-50e_lvis_federated_clip.py'


# model settings
model = dict(
    backbone=dict(
        model_name='RN50',
        pretrained='openai'
    )
)
