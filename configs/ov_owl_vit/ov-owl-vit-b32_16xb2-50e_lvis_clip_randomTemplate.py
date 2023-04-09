from mmdet.utils.clip import CLIP_BEST_PROMPT_TEMPLATES, TRAINING_PROMPT_TEMPLATES

_base_ = 'ov-owl-vit-b32_16xb2-50e_lvis_clip.py'

assert _base_.train_dataloader.dataset.pipeline == _base_.train_pipeline
assert _base_.train_pipeline[-2].type == 'ClipTokenizeQueries'
_base_.train_pipeline[-2]['templates'] = TRAINING_PROMPT_TEMPLATES
_base_.train_pipeline[-2]['canonicalize'] = True
_base_.train_pipeline[-2]['mode'] = 'random'
_base_.train_dataloader.dataset.pipeline = _base_.train_pipeline

assert _base_.test_dataloader.dataset.pipeline == _base_.test_pipeline
assert _base_.val_dataloader.dataset.pipeline == _base_.test_pipeline
assert _base_.test_pipeline[-2].type == 'ClipTokenizeQueries'
_base_.test_pipeline[-2]['templates'] = CLIP_BEST_PROMPT_TEMPLATES
_base_.test_pipeline[-2]['canonicalize'] = True
_base_.test_pipeline[-2]['mode'] = 'ensemble'

_base_.test_dataloader.dataset.pipeline = _base_.test_pipeline
_base_.val_dataloader.dataset.pipeline = _base_.test_pipeline
