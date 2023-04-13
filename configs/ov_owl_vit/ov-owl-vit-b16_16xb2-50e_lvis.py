_base_ = 'ov-owl-vit-b32_16xb2-50e_lvis.py'

model_name = 'ViT-B-16'
pretrained = 'openai'

model = dict(
    backbone=dict(
        model_name=model_name,
        pretrained=pretrained))

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False)
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False)
test_dataloader = val_dataloader
