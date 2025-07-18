"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import importlib.metadata

from torch import Tensor

if "0.15.2" in importlib.metadata.version("torchvision"):
    import torchvision

    torchvision.disable_beta_transforms_warning()

    from torchvision.datapoints import BoundingBox as BoundingBoxes
    from torchvision.datapoints import BoundingBoxFormat, Image, Mask, Video
    from torchvision.transforms.v2 import SanitizeBoundingBox as SanitizeBoundingBoxes

    _boxes_keys = ["format", "spatial_size"]

elif "0.17" > importlib.metadata.version("torchvision") >= "0.16":
    import torchvision

    torchvision.disable_beta_transforms_warning()

    from torchvision.transforms.v2 import SanitizeBoundingBoxes
    from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Image, Mask, Video

    _boxes_keys = ["format", "canvas_size"]

elif importlib.metadata.version("torchvision") >= "0.17":
    import torchvision
    from torchvision.transforms.v2 import SanitizeBoundingBoxes
    from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Image, Mask, Video

    _boxes_keys = ["format", "canvas_size"]

else:
    raise RuntimeError("Please make sure torchvision version >= 0.15.2")


def convert_to_tv_tensor(tensor: Tensor, key: str, box_format="xyxy", spatial_size=None) -> Tensor:
    """
    Args:
        tensor (Tensor): input tensor
        key (str): transform to key

    Return:
        Dict[str, TV_Tensor]
    """
    assert key in (
        "boxes",
        "masks",
    ), "Only support 'boxes' and 'masks'"

    if key == "boxes":
        box_format = getattr(BoundingBoxFormat, box_format.upper())
        _kwargs = dict(zip(_boxes_keys, [box_format, spatial_size]))
        return BoundingBoxes(tensor, **_kwargs)

    if key == "masks":
        return Mask(tensor)
