from configs._base_.datasets.coco_detection import dataset_type, data_root, train_pipeline

_base_ = [
    './retinanet_r50_fpn_1x_coco.py'
]

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='MixUpDataset',
        force_alpha=1.0,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='annotations/instances_train2017.json',
            data_prefix=dict(img='train2017/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline)))

# model settings
model = dict(
    train_cfg=dict(
        assigner=dict(
            _delete_=True,
            type='SoftIoUAssigner',
            num_classes=80,
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0)
    ))
