_base_ = [
    './retinanet_r50_fpn_1x_lvis-v1.py'
]

# model settings
model = dict(
    bbox_head=dict(
        retina_cls=dict(
            type='ClipConvClassPredictor'
        )
    )
)