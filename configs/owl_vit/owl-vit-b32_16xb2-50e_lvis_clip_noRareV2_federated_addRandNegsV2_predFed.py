_base_ = 'owl-vit-b32_16xb2-50e_lvis_clip_noRareV2_federated_addRandNegsV2.py'

assert _base_.test_dataloader.dataset.pipeline == _base_.test_pipeline
assert _base_.val_dataloader.dataset.pipeline == _base_.test_pipeline
_base_.test_pipeline = _base_.test_pipeline[:-1] + [
    dict(type='AddMissingKeys', pos_label_ids=list(range(1203)), neg_label_ids=[], not_exhaustive_label_ids=[])] + _base_.test_pipeline[-1:]
_base_.test_pipeline[-1].meta_keys = (
    'img_id', 'img_path', 'ori_shape', 'img_shape',
    'scale_factor', 'neg_label_ids', 'not_exhaustive_label_ids', 'pos_label_ids')
_base_.test_dataloader.dataset.pipeline = _base_.test_pipeline
_base_.val_dataloader.dataset.pipeline = _base_.test_pipeline


model = dict(
    bbox_head=dict(
        test_cfg=dict(
            use_category_ids=True
        )))
