_base_ = 'ov-owl-vit-b32_16xb2-50e_lvis_clip.py'

assert _base_.train_dataloader.dataset.pipeline == _base_.train_pipeline
assert _base_.train_pipeline[2].type == 'RandomChoiceResize', _base_.train_pipeline
_base_.train_pipeline[2].scales = [(w//2, h//2) for w, h in _base_.train_pipeline[2].scales]
_base_.train_dataloader.dataset.pipeline = _base_.train_pipeline

assert _base_.test_dataloader.dataset.pipeline == _base_.test_pipeline
assert _base_.val_dataloader.dataset.pipeline == _base_.test_pipeline
assert _base_.test_pipeline[1].type == 'Resize', _base_.test_pipeline
_base_.test_pipeline[1].scale = tuple(d//2 for d in _base_.test_pipeline[1].scale)
_base_.test_dataloader.dataset.pipeline = _base_.test_pipeline
_base_.val_dataloader.dataset.pipeline = _base_.test_pipeline
