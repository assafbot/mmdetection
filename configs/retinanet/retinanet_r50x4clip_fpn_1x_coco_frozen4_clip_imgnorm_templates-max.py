from mmengine.model import bias_init_with_prob

_base_ = './retinanet_r50x4clip_fpn_1x_coco_frozen4_clip_imgnorm_templates-avg.py'

# model settings
model = dict(
    bbox_head=dict(
        retina_cls=dict(
            reduction='max'
        )
    )
)