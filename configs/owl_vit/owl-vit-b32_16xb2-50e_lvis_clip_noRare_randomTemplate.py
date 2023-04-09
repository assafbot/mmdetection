_base_ = 'owl-vit-b32_16xb2-50e_lvis_clip_noRare.py'

model = dict(
    bbox_head=dict(
        fc_cls=dict(
            class_embeddings='mentee://mmdetection/embeddings/LVISV1Dataset_classes1203_ViT-B-32_openai_normalize_embeddings_templates81.pth'
        )))