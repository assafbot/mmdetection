_base_ = 'ov-owl-r50_16xb2-50e_lvis.py'

model_name = 'RN101'
pretrained = 'openai'

model = dict(
    backbone=dict(
        model_name=model_name,
        pretrained=pretrained,
        init_cfg=dict(type='Pretrained', checkpoint='mentee://mmdetection/pretrained/RN101_openai.pth')))
