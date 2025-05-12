'''
Dome-DETR: Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
'''

import torch
from src.zoo.dome.box_ops import box_iou

def dynamic_nms(boxes, scores, classes, iou_thresholds):
    unique_classes = classes.unique()
    keep_mask = torch.zeros_like(classes, dtype=torch.bool)
    for cls in unique_classes:
        cls_mask = (classes == cls)
        boxes_cls = boxes[cls_mask]
        scores_cls = scores[cls_mask]
        thresholds_cls = iou_thresholds[cls_mask]
        keep_cls = _per_class_dynamic_nms(boxes_cls, scores_cls, thresholds_cls)
        cls_indices = torch.nonzero(cls_mask, as_tuple=True)[0]
        keep_mask[cls_indices[keep_cls]] = True
    return torch.nonzero(keep_mask, as_tuple=True)[0]

def _per_class_dynamic_nms(boxes, scores, iou_thresholds):
    keep = []
    idxs = scores.argsort(descending=True)
    while idxs.numel() > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size(0) == 1:
            break
        ious, _ = box_iou(boxes[i].unsqueeze(0), boxes[idxs[1:]])
        ious = ious.squeeze(0)
        suppress = (ious >= iou_thresholds[i])
        idxs = idxs[1:][~suppress]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def dynamic_nms_fast(boxes, scores, classes, iou_thresholds):
    unique_classes = classes.unique()
    keep_mask = torch.zeros_like(classes, dtype=torch.bool)
    for cls in unique_classes:
        cls_mask = (classes == cls)
        boxes_cls = boxes[cls_mask]
        scores_cls = scores[cls_mask]
        thresholds_cls = iou_thresholds[cls_mask]
        keep_cls = _per_class_dynamic_nms_vectorized(boxes_cls, scores_cls, thresholds_cls)
        cls_indices = torch.nonzero(cls_mask, as_tuple=True)[0]
        keep_mask[cls_indices[keep_cls]] = True
    return torch.nonzero(keep_mask, as_tuple=True)[0]

def _per_class_dynamic_nms_vectorized(boxes, scores, iou_thresholds):
    # 按分数降序排列
    order = scores.argsort(descending=True)
    boxes = boxes[order]
    thresholds = iou_thresholds[order]
    
    # 预计算所有框之间的 IoU 矩阵（对称矩阵）
    iou_matrix, _ = box_iou(boxes, boxes)
    
    num = boxes.shape[0]
    keep_flags = torch.ones(num, dtype=torch.bool, device=boxes.device)
    keep = []
    for i in range(num):
        if not keep_flags[i]:
            continue
        keep.append(i)
        # 对于排序后位于 i 后面的所有框，如果 IoU 大于等于当前框的动态阈值，则置为 False
        # 注意：这里仅比较后面的框，避免重复判断
        if i < num - 1:
            # mask 指示第 i 框与后续各框的 IoU 是否超过阈值 thresholds[i]
            mask = iou_matrix[i, (i+1):] >= thresholds[i]
            keep_flags[(i+1):] &= ~mask
    # 返回在原始排序中的索引，再映射回原始索引
    return order[torch.tensor(keep, device=boxes.device)]
