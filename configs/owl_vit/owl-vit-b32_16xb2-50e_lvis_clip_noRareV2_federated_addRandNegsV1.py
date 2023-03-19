_base_ = 'owl-vit-b32_16xb2-50e_lvis_clip_noRareV2_federated.py'

assert _base_.train_dataloader.dataset.pipeline == _base_.train_pipeline
assert _base_.train_pipeline[-2].type == 'RemoveLVISRareLabels'  # make sure we add negative before rare labels are removed
_base_.train_pipeline = _base_.train_pipeline[:-2] + [
    dict(type='AddRandomNegatives', num_classes=_base_.model.bbox_head.num_classes, total=50)] \
                        + _base_.train_pipeline[-2:]
_base_.train_dataloader.dataset.pipeline = _base_.train_pipeline


