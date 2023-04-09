# dataset settings
dataset_type = 'OpenImagesDataset'
data_root = '/data/openimages/'

num_queries = num_classes = 601

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
                    'not_exhaustive_label_ids',  # 'neg_label_ids', 'pos_label_ids',
                    'query_mapping'))
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='AddMissingKeys', pos_label_ids=[],
         image_level_labels=list(range(num_classes)),
         not_exhaustive_label_ids=[]),
    dict(type='AddQuerySet', num_queries=None),
    # TODO: find a better way to collect image_level_labels
    dict(
        type='PackDetInputs', additional_input_keys=['query'],
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'instances', 'image_level_labels',
                   # 'neg_label_ids', 'not_exhaustive_label_ids', 'pos_label_ids',
                   'query_mapping'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=0,  # workers_per_gpu > 0 may occur out of memory
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations_v6/oidv6-train-annotations-bbox.csv',
        data_prefix=dict(img='train/'),
        label_file='annotations_v6/class-descriptions-boxable.csv',
        hierarchy_file='annotations_v6/bbox_labels_600_hierarchy.json',
        meta_file='annotations_v6/train-image-metas.pkl',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations_v6/validation-annotations-bbox.csv',
        data_prefix=dict(img='validation/'),
        label_file='annotations_v6/class-descriptions-boxable.csv',
        hierarchy_file='annotations_v6/bbox_labels_600_hierarchy.json',
        meta_file='annotations_v6/validation-image-metas.pkl',
        image_level_ann_file='annotations_v6/validation-'
                             'annotations-human-imagelabels-boxable.csv',
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='OpenImagesMetric',
    iou_thrs=0.5,
    ioa_thrs=0.5,
    use_group_of=True,
    get_supercategory=True)
test_evaluator = val_evaluator
