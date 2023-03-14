_base_ = [
    './retinanet_clip-vit-b32.py',
]

# model settings
model = dict(
    backbone=dict(model_name='ViT-B-16'),
    bbox_head=dict(anchor_generator=dict(strides=[16]))
)
