from mmengine.model import bias_init_with_prob

_base_ = './retinanet_r50x4clip_fpn_1x_lvis-v1_frozen4.py'

# model settings
model = dict(
    bbox_head=dict(
        retina_cls=dict(
            type='ClipConvClassPredictor',
            norm_image=True,
            model_name='RN50x4',
            pretrained='openai',
            init_cfg=[dict(type='Normal', layer='Conv2d', std=0.01),
                      dict(type='Constant', layer='ScaleLayer', val=10, bias=bias_init_with_prob(0.01))]
        )
    )
)