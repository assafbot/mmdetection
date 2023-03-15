from mmengine.model import bias_init_with_prob

_base_ = 'deformable-detr_r50v2_16xb2-50e_coco_frozen.py'

# model settings
model = dict(
    bbox_head=dict(
        fc_cls=dict(
            type='ClipLinearClassPredictor',
            norm_image=True,
            model_name='RN50',
            pretrained='openai',
            init_cfg=[dict(type='Normal', layer='Conv2d', std=0.01),
                      dict(type='Constant', layer='ScaleLayer', val=10, bias=bias_init_with_prob(0.01))]
        )
    )
)
