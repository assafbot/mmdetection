_base_ = 'ov-owl-convnext_16xb2-50e_lvis.py'

assert _base_.train_dataloader.dataset.pipeline == _base_.train_pipeline
_base_.train_pipeline = _base_.train_pipeline[:-1] + [
    dict(type='ClipTokenizeQueries', model_name=_base_.model_name)] + _base_.train_pipeline[-1:]
_base_.train_dataloader.dataset.pipeline = _base_.train_pipeline

assert _base_.test_dataloader.dataset.pipeline == _base_.test_pipeline
assert _base_.val_dataloader.dataset.pipeline == _base_.test_pipeline
_base_.test_pipeline = _base_.test_pipeline[:-1] + [
    dict(type='ClipTokenizeQueries', model_name=_base_.model_name)] + _base_.test_pipeline[-1:]
_base_.test_dataloader.dataset.pipeline = _base_.test_pipeline
_base_.val_dataloader.dataset.pipeline = _base_.test_pipeline

model = dict(
    bbox_head=dict(
        fc_cls=dict(
            _delete_=True,
            type='QueryClipLinearClassPredictor',
            model_name=_base_.model_name,
            pretrained=_base_.pretrained)))

optim_wrapper = dict(
    paramwise_cfg=dict(
        _delete_=True,
        custom_keys=dict({
            'backbone': dict(lr_mult=0.1),
            'bbox_head.fc_cls.model': dict(lr_mult=0.01)
        })))