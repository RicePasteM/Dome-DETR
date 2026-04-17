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
        cls_indices = torch.nonzero(cls_mask).squeeze(1)
        keep_mask[cls_indices[keep_cls]] = True
    return torch.nonzero(keep_mask).squeeze(1)

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


@torch.jit.script
def _per_class_dynamic_nms_vectorized(boxes, scores, iou_thresholds):
    # 按分数降序排列
    order = scores.argsort(descending=True)
    boxes = boxes[order]
    thresholds = iou_thresholds[order]
    
    # 预计算所有框之间的 IoU 矩阵（对称矩阵）
    iou_matrix, _ = box_iou(boxes, boxes)
    
    num = boxes.shape[0]
    keep_flags = torch.ones(num, dtype=torch.bool, device=boxes.device)
    keep_indicator = torch.zeros(num, dtype=torch.bool, device=boxes.device)
    for i in range(num):
        current_keep = keep_flags[i]
        keep_indicator[i] = current_keep
        # 对于排序后位于 i 后面的所有框，如果 IoU 大于等于当前框的动态阈值，则置为 False。
        # 空切片会自然成为 no-op，这样可以避免导出控制流 If。
        mask = iou_matrix[i, (i + 1):] >= thresholds[i]
        keep_flags[(i + 1):] = torch.logical_and(
            keep_flags[(i + 1):],
            torch.logical_not(torch.logical_and(mask, current_keep)),
        )

    keep = torch.nonzero(keep_indicator).squeeze(1)
    # 返回在原始排序中的索引，再映射回原始索引
    return order[keep]


@torch.jit.script
def _offset_boxes_by_class(boxes, classes):
    # Decoder boxes are normalized xyxy boxes derived from sigmoid(cx, cy, w, h),
    # so coordinates stay within [-0.5, 1.5]. A stride of 4.0 cleanly separates
    # classes while preserving exact class-wise NMS behavior.
    offsets = classes.to(boxes.dtype).unsqueeze(1) * 4.0
    return boxes + offsets


@torch.jit.script
def dynamic_nms_fast(boxes, scores, classes, iou_thresholds):
    # Class-specific coordinate offsets make cross-class IoU exactly zero,
    # which preserves the original class-wise NMS behavior without a Unique op.
    boxes_offset = _offset_boxes_by_class(boxes, classes)
    keep_idx = _per_class_dynamic_nms_vectorized(boxes_offset, scores, iou_thresholds)
    return torch.sort(keep_idx).values


def dynamic_nms_fast_static(boxes, scores, classes, iou_thresholds, active_mask, max_candidates: int):
    """
    Export-only exact greedy NMS with a fixed Python loop count so ONNX does not
    contain a Loop op. `boxes` must already have a static candidate count.
    """
    boxes_offset = _offset_boxes_by_class(boxes, classes)
    order = scores.argsort(descending=True)
    boxes_sorted = boxes_offset.index_select(0, order)
    thresholds_sorted = iou_thresholds.index_select(0, order)
    active_sorted = active_mask.index_select(0, order)

    iou_matrix, _ = box_iou(boxes_sorted, boxes_sorted)
    keep_flags = active_sorted.clone()
    keep_indicator = torch.zeros_like(active_sorted)

    for i in range(max_candidates):
        current_keep = keep_flags[i]
        keep_indicator[i] = current_keep
        if i + 1 < max_candidates:
            suppress = iou_matrix[i, (i + 1):] >= thresholds_sorted[i]
            keep_flags[(i + 1):] = torch.logical_and(
                keep_flags[(i + 1):],
                torch.logical_not(torch.logical_and(suppress, current_keep)),
            )

    keep_sorted = torch.nonzero(keep_indicator).squeeze(1)
    return torch.sort(order.index_select(0, keep_sorted)).values
