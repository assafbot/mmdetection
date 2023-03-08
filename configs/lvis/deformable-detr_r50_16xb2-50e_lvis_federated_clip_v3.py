_base_ = 'deformable-detr_r50_16xb2-50e_lvis_federated_clip_v2.py'

model = dict(
    bbox_head=dict(
        fc_cls=dict(random_template=True,
                    scale_with_input=True,
                    init_cfg=[dict(type='Normal', layer='Conv2d', std=1e-6)])
    )
)
