_base_ = 'ov-owl-vit-b32_16xb2-50e_lvis_clip.py'

train_dataloader = dict(
    sampler=dict(
        _delete_=True,
        type='FiniteSampler',
        length=12424 * 4 * 2),  # 12424 iterations * 4 GPUs * 2 batch size
    dataset=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=_base_.train_dataloader.dataset))