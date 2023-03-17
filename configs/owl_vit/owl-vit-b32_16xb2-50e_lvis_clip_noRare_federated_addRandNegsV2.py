_base_ = 'owl-vit-b32_16xb2-50e_lvis_clip_noRare_federated.py'

assert _base_.train_dataloader.dataset.pipeline == _base_.train_pipeline
_base_.train_pipeline = _base_.train_pipeline[:-1] + [
    dict(type='AddRandomNegativesV2', num_classes=_base_.model.bbox_head.num_classes, total=50)] \
                        + _base_.train_pipeline[-1:]
_base_.train_dataloader.dataset.pipeline = _base_.train_pipeline


