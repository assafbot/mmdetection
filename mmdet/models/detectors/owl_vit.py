# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from typing import List, Tuple, Union

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.base import BaseDetector


@MODELS.register_module()
class OWLViT(BaseDetector, metaclass=ABCMeta):
    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # init model layers
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Union[dict, list]:
        img_feats = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(img_feats, batch_data_samples=batch_data_samples)
        return losses

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True) -> SampleList:
        img_feats = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(img_feats, rescale=rescale, batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results_list)
        return batch_data_samples

    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        img_feats = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(img_feats)
        return results

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x
