from mmengine.model import bias_init_with_prob

_base_ = './retinanet_r50x4clip_fpn_1x_coco_frozen4_clip_imgnorm.py'

# model settings
model = dict(
    bbox_head=dict(
        retina_cls=dict(
            reduction='avg',
            templates=[
                '{}',
                'a close-up photo of a {}.',
                'a photo of the {}.',
                'a photo of one {}.',
                'a photo of a {}.',
                'a photo of a small {}.',
                'a photo of a large {}.',
                'a photo of the small {}.',
                'a photo of the large {}.',
            ]
        )
    )
)