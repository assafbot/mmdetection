import open_clip
import torch
from tqdm import tqdm

from mmdet.datasets import LVISV1Dataset
from mmdet.models.dense_heads.class_predictors import _canonicalize
from mmdet.utils.clip import TRAINING_PROMPT_TEMPLATES

class_names = LVISV1Dataset.METAINFO['classes']
templates = TRAINING_PROMPT_TEMPLATES
model_name = 'RN50'
model = open_clip.create_model(model_name, 'openai')
model.eval()
tokenizer = open_clip.get_tokenizer(model_name)

with torch.no_grad():
    class_embeddings = []
    for template in tqdm(templates):
        texts = [template.format(class_name) for class_name in class_names]
        texts = _canonicalize(texts)
        class_embeddings.append((model.encode_text(tokenizer(texts)).T)[None])

    class_embeddings = torch.cat(class_embeddings)
