"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

from typing import Callable, Optional

import torchvision

from ...core import register


@register()
class CIFAR10(torchvision.datasets.CIFAR10):
    __inject__ = ["transform", "target_transform"]

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)
