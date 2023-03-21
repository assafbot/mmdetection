# dataset settings
from mmdet.datasets import LVISV1Dataset

dataset_type = 'LVISV05Dataset'
data_root = '/data/datasets/lvis/'

num_queries = 100
num_classes = 1203

# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                (1333, 768), (1333, 800)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='AddRandomNegativesV2', num_classes=num_classes, total=50),
    dict(type='RemoveLVISRareLabels', remove_labels=True),
    dict(type='AddQuerySet', num_queries=num_queries),
    dict(type='PackDetInputs', additional_input_keys=['query'],
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction',
                    # 'neg_label_ids', 'not_exhaustive_label_ids', 'pos_label_ids',
                    'query_mapping'))
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='AddMissingKeys', pos_label_ids=list(range(num_classes)),
         neg_label_ids=[],
         not_exhaustive_label_ids=[],
         metainfo=LVISV1Dataset.METAINFO),
    dict(type='AddQuerySet', num_queries=None),
    dict(type='PackDetInputs', additional_input_keys=['query'],
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
                    # 'neg_label_ids', 'not_exhaustive_label_ids', 'pos_label_ids',
                    'query_mapping'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/lvis_v0.5_train.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/lvis_v0.5_val.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='LVISMetric',
    ann_file=data_root + 'annotations/lvis_v0.5_val.json',
    metric='bbox')
test_evaluator = val_evaluator
