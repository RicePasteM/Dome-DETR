"""
Copied from D-FINE (https://github.com/Peterande/D-FINE)
Copyright(c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import torch
import torch.nn as nn

from src.core import YAMLConfig


def main(
    args,
):
    """main"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print("not load model.state_dict, use default init state dict...")

    eval_size = cfg.yaml_cfg.get("eval_spatial_size", None)
    if args.input_size is not None:
        input_h = input_w = args.input_size
    elif eval_size is not None:
        input_h, input_w = eval_size
    else:
        input_h = input_w = 640

    class Model(nn.Module):
        def __init__(
            self,
        ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().eval()

    data = torch.rand(args.batch_size, 3, input_h, input_w)
    size = torch.tensor([[input_h, input_w]]).repeat(args.batch_size, 1)
    with torch.no_grad():
        _ = model(data, size)

    output_file = args.resume.replace(".pth", ".onnx") if args.resume else "model.onnx"

    torch.onnx.export(
        model,
        (data, size),
        output_file,
        input_names=["images", "orig_target_sizes"],
        output_names=["labels", "boxes", "scores"],
        dynamic_axes=None,
        opset_version=16,
        verbose=False,
        do_constant_folding=True,
    )

    if args.check:
        import onnx

        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print("Check export onnx model done...")

    if args.simplify:
        import onnx
        import onnxsim

        input_shapes = {"images": data.shape, "orig_target_sizes": size.shape}
        onnx_model_simplify, check = onnxsim.simplify(output_file, test_input_shapes=input_shapes)
        onnx.save(onnx_model_simplify, output_file)
        print(f"Simplify onnx model {check}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="configs/dome/dfine_hgnetv2_l_coco.yml",
        type=str,
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Square input size. Defaults to eval_spatial_size from the config when available.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Export batch size. Batch > 1 is not recommended for the current deploy path.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    main(args)
