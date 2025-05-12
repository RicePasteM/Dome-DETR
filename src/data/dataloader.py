"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import random
from functools import partial

import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms.v2 as VT
from torch.utils.data import default_collate
from torchvision.transforms.v2 import InterpolationMode
from torchvision.transforms.v2 import functional as VF

from ..core import register

torchvision.disable_beta_transforms_warning()


__all__ = [
    "DataLoader",
    "BaseCollateFunction",
    "BatchImageCollateFunction",
    "batch_image_collate_fn",
]


@register()
class DataLoader(data.DataLoader):
    __inject__ = ["dataset", "collate_fn"]

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ["dataset", "batch_size", "num_workers", "drop_last", "collate_fn"]:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string

    def set_epoch(self, epoch):
        self._epoch = epoch
        self.dataset.set_epoch(epoch)
        self.collate_fn.set_epoch(epoch)

    @property
    def epoch(self):
        return self._epoch if hasattr(self, "_epoch") else -1

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        assert isinstance(shuffle, bool), "shuffle must be a boolean"
        self._shuffle = shuffle


@register()
def batch_image_collate_fn(items):
    """only batch image"""
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]


class BaseCollateFunction(object):
    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, "_epoch") else -1

    def __call__(self, items):
        raise NotImplementedError("")


# def generate_scales(base_size, base_size_repeat):

#     """修改后的 generate_scales 函数支持长方形尺寸"""
#     if isinstance(base_size, list) or isinstance(base_size, tuple):
#         h, w = base_size
#         h_scales = []
#         w_scales = []
        
#         # 为高度生成尺度
#         h_repeat = (h - int(h * 0.75 / 32) * 32) // 32
#         h_scales = [int(h * 0.75 / 32) * 32 + i * 32 for i in range(h_repeat)]
#         h_scales += [h] * base_size_repeat
#         h_scales += [int(h * 1.25 / 32) * 32 - i * 32 for i in range(h_repeat)]
        
#         # 为宽度生成尺度
#         w_repeat = (w - int(w * 0.75 / 32) * 32) // 32
#         w_scales = [int(w * 0.75 / 32) * 32 + i * 32 for i in range(w_repeat)]
#         w_scales += [w] * base_size_repeat
#         w_scales += [int(w * 1.25 / 32) * 32 - i * 32 for i in range(w_repeat)]
        
#         return list(zip(h_scales, w_scales))
#     else:
#         # 保持原有的方形图片处理逻辑
#         scale_repeat = (base_size - int(base_size * 0.75 / 32) * 32) // 32
#         scales = [int(base_size * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)]
#         scales += [base_size] * base_size_repeat
#         scales += [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
#         return [(s, s) for s in scales]


# @register()
# class BatchImageCollateFunction(BaseCollateFunction):
#     def __init__(
#         self,
#         stop_epoch=None,
#         ema_restart_decay=0.9999,
#         base_size=(640, 640),
#         base_size_repeat=None,
#     ) -> None:
#         super().__init__()
#         self.base_size = base_size
#         if isinstance(self.base_size, str):
#             try:
#                 self.base_size = int(self.base_size)
#             except ValueError:
#                 try:
#                     self.base_size = self.base_size.strip('()[]')
#                     self.base_size = tuple(int(x.strip()) for x in self.base_size.split(','))
#                 except:
#                     raise ValueError(f"Cannot convert base_size string '{self.base_size}' to valid size format")
#         self.scales = (
#             generate_scales(self.base_size, base_size_repeat) if base_size_repeat is not None else None
#         )
#         self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
#         self.ema_restart_decay = ema_restart_decay

#     def __call__(self, items):
#         images = torch.cat([x[0][None] for x in items], dim=0)
#         targets = [x[1] for x in items]

#         if self.scales is not None and self.epoch < self.stop_epoch:
#             sz = random.choice(self.scales)  # 现在 sz 是 (height, width) 元组
#             images = F.interpolate(images, size=sz)
#             if "masks" in targets[0]:
#                 for tg in targets:
#                     tg["masks"] = F.interpolate(tg["masks"], size=sz, mode="nearest")

#         return images, targets


def generate_scales(base_size, base_size_repeat, window_size):
    """修改后的generate_scales函数确保处理后尺寸满足(size/4)%window_size==0"""
    if isinstance(base_size, (list, tuple)):
        h, w = base_size
        step = 4 * window_size
        # 调整h和w为step的倍数
        h = (h // step) * step or step
        w = (w // step) * step or step

        # 生成h的scales
        h_min = int(h * 0.75)
        start_low_h = ((h_min + step - 1) // step) * step
        low_to_base_h = list(range(start_low_h, h + 1, step))
        h_max = int(h * 1.25)
        end_high_h = (h_max // step) * step
        high_part_h = list(range(end_high_h, h - 1, -step)) if end_high_h >= h + step else []
        h_scales = low_to_base_h + [h] * base_size_repeat + high_part_h

        # 生成w的scales
        w_min = int(w * 0.75)
        start_low_w = ((w_min + step - 1) // step) * step
        low_to_base_w = list(range(start_low_w, w + 1, step))
        w_max = int(w * 1.25)
        end_high_w = (w_max // step) * step
        high_part_w = list(range(end_high_w, w - 1, -step)) if end_high_w >= w + step else []
        w_scales = low_to_base_w + [w] * base_size_repeat + high_part_w

        # 生成笛卡尔积组合
        scales = []
        for h_scale in h_scales:
            for w_scale in w_scales:
                scales.append((h_scale, w_scale))
        return scales
    else:
        # 处理方形尺寸
        step = 4 * window_size
        base_size = (base_size // step) * step or step
        base_min = int(base_size * 0.75)
        start_low = ((base_min + step - 1) // step) * step
        low_to_base = list(range(start_low, base_size + 1, step))
        base_max = int(base_size * 1.25)
        end_high = (base_max // step) * step
        high_part = list(range(end_high, base_size - 1, -step)) if end_high >= base_size + step else []
        scales = low_to_base + [base_size] * base_size_repeat + high_part
        return [(s, s) for s in scales]

@register()
class BatchImageCollateFunction(BaseCollateFunction):
    def __init__(
        self,
        stop_epoch=None,
        ema_restart_decay=0.9999,
        base_size=(640, 640),
        base_size_repeat=None,
        mwas_window_size=20,  # 新增window_size参数
    ) -> None:
        super().__init__()
        self.base_size = base_size
        if isinstance(self.base_size, str):
            try:
                self.base_size = int(self.base_size)
            except ValueError:
                try:
                    self.base_size = self.base_size.strip('()[]')
                    self.base_size = tuple(int(x.strip()) for x in self.base_size.split(','))
                except:
                    raise ValueError(f"Cannot convert base_size string '{self.base_size}' to valid size format")
        self.window_size = mwas_window_size  # 初始化window_size
        self.scales = (
            generate_scales(self.base_size, base_size_repeat, self.window_size) 
            if base_size_repeat is not None 
            else None
        )
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000
        self.ema_restart_decay = ema_restart_decay

    def __call__(self, items):
        images = torch.cat([x[0][None] for x in items], dim=0)
        targets = [x[1] for x in items]

        if self.scales is not None and self.epoch < self.stop_epoch:
            sz = random.choice(self.scales)
            images = F.interpolate(images, size=sz)
            if "masks" in targets:
                for tg in targets:
                    tg["masks"] = F.interpolate(tg["masks"], size=sz, mode="nearest")

        return images, targets