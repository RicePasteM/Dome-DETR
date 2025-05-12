"""
Dome-DETR: Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch

from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .utils import inverse_sigmoid


def get_contrastive_denoising_training_group(
    targets,
    num_classes,
    num_queries,
    class_embed,
    num_denoising=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
    batch_queries_num=None,
    num_heads=8,
):
    """cnd"""
    if num_denoising <= 0:
        return None, None, None, None

    num_gts = [len(t["labels"]) for t in targets]
    device = targets[0]["labels"].device

    # pad gt to max_num of a batch
    bs = len(num_gts)

    max_gt_num = max(num_gts)

    if max_gt_num == 0:
        # 构造空张量确保class_embed被调用
        input_query_class = torch.full([bs, 0], num_classes, dtype=torch.int32, device=device)
        input_query_logits = class_embed(input_query_class)  # 触发参数计算图
        input_query_bbox = torch.zeros([bs, 0, 4], device=device)
        input_query_bbox_unact = inverse_sigmoid(input_query_bbox)
        # 构建仅包含匹配查询的注意力掩码
        tgt_size = 0 + num_queries
        attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool, device=device)
        dn_meta = {
            "dn_positive_idx": None, 
            "dn_num_group": 0, 
            "dn_num_split": [0, num_queries]
        }
        return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta

    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group

    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)

    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]["labels"]
            input_query_bbox[i, :num_gt] = targets[i]["boxes"]
            pad_gt_mask[i, :num_gt] = 1
    # each group has positive and negative queries.
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)

    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        # randomly put a new one here
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        # shrink_mask = torch.zeros_like(rand_sign)
        # shrink_mask[:, :, :2] = (rand_sign[:, :, :2] == 1)  # rand_sign == 1 → (x1, y1) ↘ →  smaller bbox
        # shrink_mask[:, :, 2:] = (rand_sign[:, :, 2:] == -1)  # rand_sign == -1 →  (x2, y2) ↖ →  smaller bbox
        # mask = rand_part > (upper_bound / (upper_bound+1))
        # # this is to make sure the dn bbox can be reversed to the original bbox by dome head.
        # rand_sign = torch.where((shrink_mask * (1 - negative_gt_mask) * mask).bool(), \
        #                         rand_sign * upper_bound / (upper_bound+1) / rand_part, rand_sign)
        known_bbox += rand_sign * rand_part * diff
        known_bbox = torch.clip(known_bbox, min=0.0, max=1.0)
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        input_query_bbox[input_query_bbox < 0] *= -1
        input_query_bbox_unact = inverse_sigmoid(input_query_bbox)

    input_query_logits = class_embed(input_query_class)

    tgt_size = num_denoising + num_queries
    base_attn_mask = torch.full((tgt_size, tgt_size), False, device=device)

    base_attn_mask[num_denoising:, :num_denoising] = True  # 匹配查询看不到去噪部分



    # reconstruct cannot see each other
    for i in range(num_group):
            group_start = max_gt_num * 2 * i
            group_end = max_gt_num * 2 * (i + 1)
            
            # 当前组看不到后续组
            if i < num_group - 1:
                base_attn_mask[group_start:group_end, group_end:num_denoising] = True
            
            # 当前组看不到前序组
            if i > 0:
                base_attn_mask[group_start:group_end, :group_start] = True

    # 扩展为三维并添加padding屏蔽
    attn_mask = base_attn_mask.unsqueeze(0)  # [1, tgt_size, tgt_size]
    attn_mask = attn_mask.repeat(num_heads * bs, 1, 1)  # [num_heads*bs, tgt_size, tgt_size]
    
    # 添加padding屏蔽（考虑多头）
    if batch_queries_num is not None:
        for b in range(bs):
            valid_queries = batch_queries_num[b]
            padding_start = num_denoising + valid_queries
            
            if padding_start < tgt_size:
                # 计算该batch在所有头中的位置范围
                head_start = b * num_heads
                head_end = (b + 1) * num_heads
                
                # 对每个头应用相同的padding屏蔽
                attn_mask[head_start:head_end, padding_start:, :] = True
                attn_mask[head_start:head_end, :, padding_start:] = True

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries],
    }

    return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta
