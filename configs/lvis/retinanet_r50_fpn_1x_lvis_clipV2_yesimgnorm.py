from mmengine.model import bias_init_with_prob

_base_ = './retinanet_r50_fpn_1x_lvis.py'

# model settings
model = dict(
    bbox_head=dict(
        retina_cls=dict(
            type='ClipConvClassPredictor',
            norm_text=True,
            norm_image=True,
            scale=True, bias=True,
            init_cfg=[dict(type='Normal', layer='Conv2d', std=0.01),
                      dict(type='Constant', layer='ScaleLayer', val=10, bias=bias_init_with_prob(0.01))]
        )
    )
)