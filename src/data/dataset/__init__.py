"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

# from ._dataset import DetDataset
from .cifar_dataset import CIFAR10
from .coco_dataset import (
    CocoDetection,
    mscoco_category2label,
    mscoco_category2name,
    mscoco_label2category,
)
from .coco_eval_slow import CocoEvaluatorSlow
from .coco_eval import CocoEvaluator
from .coco_eval_aitod import AitodCocoEvaluator
# from .coco_eval_aitod_slow import AitodCocoEvaluatorSlow
from .coco_eval_visdrone import VisdroneCocoEvaluator
from .coco_eval_aitod_faster import AitodCocoFasterEvaluator
from .coco_utils import get_coco_api_from_dataset
from .voc_detection import VOCDetection
from .voc_eval import VOCEvaluator
