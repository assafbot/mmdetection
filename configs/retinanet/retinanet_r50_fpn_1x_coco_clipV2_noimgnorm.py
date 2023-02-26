_base_ = './retinanet_r50_fpn_1x_coco.py'

# model settings
model = dict(
    bbox_head=dict(
        retina_cls=dict(
            type='ClipConvClassPredictor',
            norm_text=True,
            norm_image=False,
            scale=1, bias_prob=0.01
        )
    )
)