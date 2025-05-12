"""
MixUp data augmentation for object detection
"""

import random
import torch
import torchvision
import torchvision.transforms.v2 as T
from PIL import Image

from ...core import register
from .._misc import convert_to_tv_tensor

torchvision.disable_beta_transforms_warning()


@register()
class MixUp(T.Transform):
    def __init__(
        self,
        alpha=1.5,
        p=0.2,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.p = p

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        image, target, dataset = inputs

        # 如果随机概率小于p，则不执行MixUp
        if random.random() > self.p:
            return image, target, dataset

        # 随机选择另一张图像
        idx = random.randint(0, len(dataset) - 1)
        mix_image, mix_target = dataset.load_item(idx)

        # 生成混合权重
        lam = random.betavariate(self.alpha, self.alpha)
        
        # 混合图像
        w, h = image.size
        mixed_image = Image.new(mode=image.mode, size=(w, h), color=0)
        mixed_image = Image.blend(image, mix_image.resize((w, h)), 1 - lam)

        # 合并标注
        mixed_target = {}
        for k in target:
            if k == "boxes":
                v1 = target[k]
                v2 = mix_target[k]
                # 确保第二张图像的框适应第一张图像的尺寸
                w2, h2 = mix_image.size
                scale_w, scale_h = w / w2, h / h2
                v2_scaled = v2.clone()
                v2_scaled[:, 0::2] *= scale_w
                v2_scaled[:, 1::2] *= scale_h
                mixed_target[k] = torch.cat([v1, v2_scaled], dim=0)
            elif k in ["labels", "area", "iscrowd"]:
                mixed_target[k] = torch.cat([target[k], mix_target[k]], dim=0)
            else:
                mixed_target[k] = target[k]

        # 添加混合权重信息
        mixed_target["mixup_weights"] = torch.tensor([lam, 1 - lam])

        return mixed_image, mixed_target, dataset 