import matplotlib
matplotlib.use('Agg')  # 设置后端为非交互式
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import io
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def visualize_detection(samples, targets, savename="output", threshold=0.5, scale_factor=1.0, return_image=False, point_mode=False, show_label=True, area_filter=None, type="xyxy"):
    print("visualizing detection")
    # 处理输入张量维度
    img_tensor = samples[0].cpu().detach() if samples.dim() == 4 else samples.cpu().detach()
    
    # 获取实际图像尺寸
    _, h, w = img_tensor.shape
    img = img_tensor.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    
    # 使用 Figure 而不是 plt 创建画布，避免全局状态
    fig = Figure(figsize=(12, 12))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.imshow(img)
    
    # 处理目标数据
    targets = targets[0] if isinstance(targets, list) else targets
    boxes = targets['boxes'].cpu().numpy()
    boxes = boxes * scale_factor


    # 过滤面积大于阈值的框
    if area_filter is not None:
        if type == "xyxy":
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        elif type == "xywh":
            areas = boxes[:, 2] * boxes[:, 3]
        mask = areas < area_filter
        boxes = boxes[mask]
        if 'labels' in targets:
            labels = targets['labels'].cpu().numpy()[mask]
        if 'scores' in targets: 
            scores = targets['scores'].cpu().numpy()[mask]


    if 'scores' in targets:
        scores = targets['scores'].cpu().numpy()
        mask = scores > threshold
        boxes = boxes[mask]
        labels = targets['labels'].cpu().numpy()[mask]
        scores = scores[mask]
    else:
        if 'labels' in targets:
            labels = targets['labels']
        else:
            labels = ["unknown"] * len(boxes)

        
    if not point_mode:
        # 绘制检测框
        for i, bl in enumerate(zip(boxes, labels)):
            box, label = bl
            if type == "xywh":
                x1, y1, w, h = box
                x1, y1 = x1 - w / 2, y1 - h / 2
                x2, y2 = x1 + w, y1 + h
            elif type == "xyxy":
                x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                            linewidth=1, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            if show_label:
                ax.text(
                    x1, y1-5, 
                    f'Class {label}' + ((" " + str(round(float(scores[i]), 3))) if ('scores' in targets) else ""), 
                    color='lime', fontsize=10, 
                    bbox=dict(facecolor='black', alpha=0.7, edgecolor='none')
            )
    else:
        for i, kps in enumerate(boxes):
            if type == "xywh":
                x, y = kps[0], kps[1]
            elif type == "xyxy":
                x, y = (kps[0] + kps[2]) / 2, (kps[1] + kps[3]) / 2
            ax.scatter(x, y, s=3, c='lime', marker='o')
    
    ax.axis('off')

    if return_image:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        buf.seek(0)
        pil_image = Image.open(buf)
        # 创建一个新的 Image 对象，这样就可以安全地关闭缓冲区
        pil_image_copy = pil_image.copy()
        pil_image.close()
        buf.close()
        return pil_image_copy
    else:
        fig.savefig(f"visualize/{savename}_rgb.png", bbox_inches='tight', dpi=300)
        
    return None