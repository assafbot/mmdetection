_base_ = 'deformable-detr_r50_16xb2-50e_coco.py'

model = dict(
    neck=dict(extra_convs_kernel=1),
    encoder=dict(num_layers=1),
    decoder=dict(num_layers=1)
)
