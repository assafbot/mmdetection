model_name = 'convnext_base'
pretrained = 'laion400m_s13b_b51k'

model = dict(
    type='OWLViT',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[122.7709383, 116.7460125, 104.09373615],
        std=[68.5005327, 66.6321579, 70.32316305],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ClipConvNext',
        model_name=model_name,
        pretrained=pretrained,
        frozen_stages=-1,
        norm_eval=True,
        init_cfg=dict(type='Pretrained', checkpoint=f'mentee://mmdetection/pretrained/{model_name}_{pretrained}.pth')),
    bbox_head=dict(
        type='OWLViTHead',
        embed_dims=1024,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))
