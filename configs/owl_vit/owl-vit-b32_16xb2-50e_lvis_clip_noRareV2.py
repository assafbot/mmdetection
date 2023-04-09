from mmengine.model import bias_init_with_prob

_base_ = 'owl-vit-b32_16xb2-50e_lvis_noRareV2.py'

model = dict(
    bbox_head=dict(
        fc_cls=dict(
            type='ClipLinearClassPredictor',
            norm_image=True,
            model_name='ViT-B-32',
            pretrained='openai',
            dataset_name='LVISV1Dataset',
            canonicalize_text_labels=True,
            init_cfg=[dict(type='Normal', layer='Conv2d', std=0.01),
                      dict(type='Constant', layer='ScaleLayer', val=10, bias=bias_init_with_prob(0.01))])))