from mmengine.model import bias_init_with_prob

_base_ = './retinanet_r50_fpn_1x_lvis-v1.py'

# model settings
model = dict(
    bbox_head=dict(
        retina_cls=dict(
            type='ClipConvClassPredictor',
            model_name='RN50',
            pretrained='openai',
            dataset_name='LVISV1Dataset'
        )
    )
)