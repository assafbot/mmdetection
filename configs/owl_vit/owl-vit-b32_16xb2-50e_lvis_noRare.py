_base_ = 'owl-vit-b32_16xb2-50e_lvis.py'


assert _base_.train_dataloader.dataset.pipeline == _base_.train_pipeline
_base_.train_pipeline = _base_.train_pipeline[:-1] + [dict(type='RemoveLVISRareLabels')] + _base_.train_pipeline[-1:]
_base_.train_dataloader.dataset.pipeline = _base_.train_pipeline
