"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch.nn as nn

from ...core import register
from .det_criterion import DetCriterion

CrossEntropyLoss = register()(nn.CrossEntropyLoss)
