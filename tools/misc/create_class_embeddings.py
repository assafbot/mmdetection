import argparse

import open_clip
import torch
from tqdm import tqdm

from mmdet.registry import DATASETS
from mmdet.utils.clip import TRAINING_PROMPT_TEMPLATES, canonicalize


def create_class_embeddings(dataset, model_name, pretrained, normalize):
    dataset = DATASETS.get(dataset)
    class_names = dataset.METAINFO['classes']
    templates = TRAINING_PROMPT_TEMPLATES
    model = open_clip.create_model(model_name, pretrained)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    with torch.no_grad():
        class_embeddings = []
        for template in tqdm(templates):
            texts = [template.format(class_name) for class_name in class_names]
            texts = canonicalize(texts)
            embs = model.encode_text(tokenizer(texts), normalize=normalize).T
            class_embeddings.append(embs[None])

        class_embeddings = torch.cat(class_embeddings).permute(1, 0, 2)

    save_name = f'/tmp/{dataset.__name__}_classes{len(class_names)}_{model_name}_{pretrained}' \
                f'_{"normalize_" if normalize else ""}embeddings_templates{len(templates)}.pth'
    torch.save(class_embeddings, save_name)
    print(f'Results saved to {save_name}')


if __name__ == '__main__':
    model_names = sorted(set(n for n, _ in open_clip.list_pretrained()))
    datasets = [n for n, d in DATASETS.module_dict.items() if hasattr(d, 'METAINFO') and 'classes' in d.METAINFO]
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('dataset', help='dataset name', choices=DATASETS.module_dict)
    parser.add_argument('model_name', help='model name', choices=model_names)
    parser.add_argument('pretrained', help='model pretrained')
    parser.add_argument('--no-normalization', help='clearml queue', action='store_true')
    args = parser.parse_args()
    normalize = not args.no_normalization

    create_class_embeddings(args.dataset, args.model_name, args.pretrained, normalize)
