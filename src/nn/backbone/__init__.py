"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

from .common import (
    FrozenBatchNorm2d,
    freeze_batch_norm2d,
    get_activation,
)
from .csp_darknet import CSPPAN, CSPDarkNet
from .csp_resnet import CSPResNet
from .hgnetv2 import HGNetv2
from .presnet import PResNet
from .test_resnet import MResNet
from .timm_model import TimmModel
from .torchvision_model import TorchVisionModel
