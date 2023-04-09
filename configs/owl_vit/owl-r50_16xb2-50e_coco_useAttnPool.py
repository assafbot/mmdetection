_base_ = 'owl-r50_16xb2-50e_coco.py'


model = dict(
    backbone=dict(use_attn_pool=True),
    bbox_head=dict(embed_dims=1024))
