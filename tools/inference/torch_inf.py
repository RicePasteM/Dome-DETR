"""
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
"""

import os
import sys
import json

import cv2  # Added for video processing
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw

INPUT_SIZE=800

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.core import YAMLConfig


def draw(images, labels, boxes, scores, thrh=0.4):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline="red")
            draw.text(
                (b[0], b[1]),
                text=f"{lab[j].item()} {round(scrs[j].item(), 2)}",
                fill="blue",
            )

        # 保存高质量图片
        im.save("torch_results.jpg", quality=95, dpi=(500, 500))


def load_coco_annotation(image_path, annotation_file):
    """加载COCO格式的标注文件"""
    # 获取图片文件名
    image_name = os.path.basename(image_path)
    
    # 读取标注文件
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # 找到对应图片的信息
    image_info = None
    for img in coco_data['images']:
        if img['file_name'] == image_name:
            image_info = img
            break
    
    if image_info is None:
        return None
    
    # 收集该图片的所有标注
    image_id = image_info['id']
    annotations = []
    for ann in coco_data['annotations']:
        if ann['image_id'] == image_id:
            annotations.append(ann)
    
    if not annotations:
        return None
        
    target = {
        'boxes': torch.tensor([ann['bbox'] for ann in annotations], dtype=torch.float32),
        'labels': torch.tensor([ann['category_id'] for ann in annotations], dtype=torch.int64),
        'image_id': torch.tensor([image_id]),
        'area': torch.tensor([ann['area'] for ann in annotations]),
        'iscrowd': torch.tensor([ann['iscrowd'] for ann in annotations])
    }

    # convert center xywh to xyxy
    target['boxes'][:, :2] = target['boxes'][:, :2] - target['boxes'][:, 2:] / 2
    target['boxes'][:, 2:] += target['boxes'][:, :2]
    
    return target


def process_image(model, device, file_path, annotation_file=None):
    im_pil = Image.open(file_path).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)

    transforms = T.Compose(
        [
            T.Resize((INPUT_SIZE, INPUT_SIZE)),
            T.ToTensor(),
        ]
    )
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    target = None
    if annotation_file:
        target = load_coco_annotation(file_path, annotation_file)
    output = model(im_data, orig_size, [target])
    labels, boxes, scores = output

    draw([im_pil], labels, boxes, scores)


def process_video(model, device, file_path):
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("torch_results.mp4", fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose(
        [
            T.Resize((INPUT_SIZE, INPUT_SIZE)),
            T.ToTensor(),
        ]
    )

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)

        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        output = model(im_data, orig_size)
        labels, boxes, scores = output

        # Draw detections on the frame
        draw([frame_pil], labels, boxes, scores)

        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'results_video.mp4'.")


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    print(cfg)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes, targets=None):
            outputs = self.model(images, targets=targets)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = args.device
    model = Model().to(device)

    # Check if the input file is an image or a video
    file_path = args.input
    if os.path.splitext(file_path)[-1].lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        # Process as image
        process_image(model, device, file_path, annotation_file=args.annotation)
        print("Image processing complete.")
    else:
        # Process as video
        process_video(model, device, file_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("-a", "--annotation", type=str, help="COCO format annotation file")
    args = parser.parse_args()
    main(args)
