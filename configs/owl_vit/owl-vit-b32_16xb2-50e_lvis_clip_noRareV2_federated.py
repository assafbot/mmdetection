_base_ = 'owl-vit-b32_16xb2-50e_lvis_clip_noRareV2.py'

assert _base_.train_dataloader.dataset.pipeline == _base_.train_pipeline
assert _base_.train_pipeline[-1].type == 'PackDetInputs'
_base_.train_pipeline = _base_.train_pipeline[:-1] + \
                        [dict(type='PackDetInputs',
                              meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
                                         'flip', 'flip_direction', 'neg_label_ids', 'not_exhaustive_label_ids',
                                         'pos_label_ids'))]
_base_.train_dataloader.dataset.pipeline = _base_.train_pipeline


model = dict(
    bbox_head=dict(
        use_category_ids=True))
