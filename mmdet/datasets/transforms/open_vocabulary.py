from random import choice
from typing import Union

import numpy as np
import open_clip
import torch
from mmcv.transforms import BaseTransform

from mmdet.datasets.transforms.loading import _filter_results
from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox.box_type import autocast_box_type


@TRANSFORMS.register_module()
class RemoveLVISRareLabels(BaseTransform):
    def __init__(self, remove_objects=True, remove_labels=False):
        self.remove_objects = remove_objects
        self.remove_labels = remove_labels

    @autocast_box_type()
    def transform(self, results: dict) -> Union[dict, None]:
        assert 'instances' in results
        instances = results['instances']
        if len(instances) == 0:
            return results

        rare_labels = results['metainfo']['rare_labels']

        if self.remove_labels:
            results['neg_label_ids'] = list(filter(lambda x: x not in rare_labels, results['neg_label_ids']))
            results['pos_label_ids'] = list(filter(lambda x: x not in rare_labels, results['pos_label_ids']))

        if self.remove_objects:
            keep = [instance['bbox_label'] not in rare_labels for instance in instances]
            results = _filter_results(results, keep)

        return results

    def __repr__(self):
        return self.__class__.__name__ + '()'


@TRANSFORMS.register_module()
class AddRandomNegatives(BaseTransform):
    def __init__(self, num_classes, total=50):
        self.num_classes = num_classes
        self.total = total

    def transform(self, results: dict) -> Union[dict, None]:
        needed = max(0, self.total - len(results['neg_label_ids']))
        rand_negs = np.random.randint(self.num_classes, size=self.total*2)
        rand_negs = [x for x in rand_negs if x not in results['pos_label_ids']]
        results['neg_label_ids'].extend(rand_negs[:needed])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(total={self.total})'


@TRANSFORMS.register_module()
class AddRandomNegativesV2(BaseTransform):
    def __init__(self, num_classes, total=50, capacity=10_000):
        self.total = total
        self.capacity = capacity
        self.num_classes = num_classes
        self.queue = np.linspace(0, self.num_classes, self.capacity+1, dtype=np.int32)[:-1]

    def transform(self, results: dict) -> Union[dict, None]:
        cand_idx = np.random.choice(self.capacity, self.total * 2)
        cand_labels = self.queue[cand_idx].tolist()
        pos_label_ids = results['pos_label_ids']
        neg_label_ids = results['neg_label_ids']
        enq_labels = pos_label_ids + neg_label_ids + cand_labels
        self.queue[cand_idx] = enq_labels[:self.total*2]

        cand_negatives = list(set(cand_labels).difference(pos_label_ids))
        np.random.shuffle(cand_negatives)

        new_negatives = neg_label_ids + cand_negatives
        new_num_negatives = max(len(neg_label_ids), self.total)
        new_negatives = new_negatives[:new_num_negatives]
        results['neg_label_ids'] = new_negatives

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(total={self.total})'


@TRANSFORMS.register_module()
class AddQuerySet(BaseTransform):
    def __init__(self, num_queries):
        self.num_queries = num_queries

    def transform(self, results: dict) -> Union[dict, None]:
        results.pop('not_exhaustive_label_ids', None)
        all_label_ids = results.pop('pos_label_ids') + results.pop('neg_label_ids')
        mapping = {v: i for i, v in enumerate(set(all_label_ids))}

        query = list(mapping)
        query = query[:self.num_queries] + [-1] * max(0, self.num_queries - len(query))
        assert len(query) == self.num_queries

        results['gt_bboxes_labels'] = np.array([mapping[l] for l in results['gt_bboxes_labels']], dtype=np.int64)
        for instance in results['instances']:
            instance['bbox_label'] = mapping[instance['bbox_label']]

        results['query'] = np.asarray(query, dtype=np.int64)

        query_mapping = torch.zeros(len(mapping), dtype=torch.int64)
        for v, i in mapping.items():
            query_mapping[i] = v
        results['query_mapping'] = query_mapping
        return results


@TRANSFORMS.register_module()
class ClipTokenizeQueries(BaseTransform):
    def __init__(self, model_name, templates=None):
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.templates = templates

    def transform(self, results: dict) -> Union[dict, None]:
        query = results['query']
        class_names = [results['metainfo']['classes'][q] if q >= 0 else '' for q in query]
        if self.templates is not None:
            class_names = [choice(self.templates).format(c) if c else c for c in class_names]

        tokens = self.tokenizer(class_names)
        results['query'] = tokens
        return results
