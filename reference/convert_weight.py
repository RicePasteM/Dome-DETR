import argparse
import os

import torch


def save_only_ema_weights(checkpoint_file):
    """Extract and save only the EMA weights."""
    checkpoint = torch.load(checkpoint_file, map_location="cpu")

    weights = {}
    if "ema" in checkpoint:
        weights["model"] = checkpoint["ema"]["module"]
    else:
        raise ValueError("The checkpoint does not contain 'ema'.")

    dir_name, base_name = os.path.split(checkpoint_file)
    name, ext = os.path.splitext(base_name)
    output_file = os.path.join(dir_name, f"{name}_converted{ext}")

    torch.save(weights, output_file)
    print(f"EMA weights saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and save only EMA weights.")
    parser.add_argument("checkpoint_file", type=str, help="Path to the input checkpoint file.")

    args = parser.parse_args()
    save_only_ema_weights(args.checkpoint_file)
