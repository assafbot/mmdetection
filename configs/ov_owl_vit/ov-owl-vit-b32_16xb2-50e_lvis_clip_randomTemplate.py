_base_ = 'ov-owl-vit-b32_16xb2-50e_lvis_clip.py'

assert _base_.train_dataloader.dataset.pipeline == _base_.train_pipeline
assert _base_.train_pipeline[-2].type == 'ClipTokenizeQueries'
_base_.train_pipeline[-2]['templates'] = [
    '{}',
    'itap of a {}.',
    'a bad photo of the {}.',
    'a origami {}.',
    'a photo of the large {}.',
    'a {} in a video game.',
    'art of the {}.',
    'a photo of the small {}.',
]
_base_.train_dataloader.dataset.pipeline = _base_.train_pipeline

assert _base_.test_dataloader.dataset.pipeline == _base_.test_pipeline
assert _base_.val_dataloader.dataset.pipeline == _base_.test_pipeline
assert _base_.test_pipeline[-2].type == 'ClipTokenizeQueries'

_base_.test_dataloader.dataset.pipeline = _base_.test_pipeline
_base_.val_dataloader.dataset.pipeline = _base_.test_pipeline

model = dict(
    bbox_head=dict(
        fc_cls=dict(
            _delete_=True,
            type='QueryClipLinearClassPredictor',
            model_name=_base_.model_name,
            pretrained=_base_.pretrained)))
