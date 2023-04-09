_base_ = 'owl-vit-b32_16xb2-50e_coco.py'

model = dict(
    bbox_head=dict(
        fc_cls=dict(
            type='ClipLinearClassPredictor',
            norm_image=True,
            model_name='ViT-B-32',
            pretrained='openai',
            dataset_name='CocoDataset',
            scale_with_input=True,
            init_cfg=[dict(type='Normal', layer='Conv2d', std=0.01)])))