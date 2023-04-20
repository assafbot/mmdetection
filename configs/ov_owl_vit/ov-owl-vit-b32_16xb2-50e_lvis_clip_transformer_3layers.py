_base_ = 'ov-owl-vit-b32_16xb2-50e_lvis_clip.py'

model = dict(
    bbox_head=dict(
        fc_cls=dict(
            type='QueryClipTransformerClassPredictor',
            num_decoder_layers=3)))
