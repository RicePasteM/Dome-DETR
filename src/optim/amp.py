"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch.cuda.amp as amp

from ..core import register

__all__ = ["GradScaler"]

GradScaler = register()(amp.grad_scaler.GradScaler)
