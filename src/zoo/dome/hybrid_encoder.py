"""
Dome-DETR: Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
from collections import OrderedDict
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from tools.visualize_src_flatten import visualize_src_flatten

# from .deformable_encoder import DeformableTransformerEncoderLayer,  DeformableTransformerEncoder
from ...core import register
from .utils import get_activation

from .defe import LiteDeFE, GaussHeatmapGenerator
from src.zoo.dome.get_roi_features import WindowProcessor, TransformerEncoder, TransformerEncoderLayer

import os
SAVE_INTERMEDIATE_VISUALIZE_RESULT = os.environ.get('SAVE_INTERMEDIATE_VISUALIZE_RESULT', 'False') == 'True'


__all__ = ["HybridEncoder"]


class ConvNormLayer_fuse(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = (
            ch_in,
            ch_out,
            kernel_size,
            stride,
            g,
            padding,
            bias,
        )

    def forward(self, x):
        if hasattr(self, "conv_bn_fused"):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, "conv_bn_fused"):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                groups=self.g,
                padding=self.padding,
                bias=True,
            )

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__("conv")
        self.__delattr__("norm")

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor()

        return kernel3x3, bias3x3

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class VGGBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act="relu"):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else act

    def forward(self, x):
        if hasattr(self, "conv"):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, "conv"):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        self.__delattr__("conv1")
        self.__delattr__("conv2")

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ELAN(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, n=2, bias=False, act="silu", bottletype=VGGBlock):
        super().__init__()
        self.c = c3
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(
            bottletype(c3 // 2, c4, act=get_activation(act)),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv3 = nn.Sequential(
            bottletype(c4, c4, act=get_activation(act)),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv4 = ConvNormLayer_fuse(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)

    def forward(self, x):
        # y = [self.cv1(x)]
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class RepNCSPELAN4(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, n=3, bias=False, act="silu"):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(
            CSPLayer(c3 // 2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv3 = nn.Sequential(
            CSPLayer(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act),
        )
        self.cv4 = ConvNormLayer_fuse(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)

    def forward_chunk(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class CSPLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=3,
        expansion=1.0,
        bias=False,
        act="silu",
        bottletype=VGGBlock,
    ):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(
            *[
                bottletype(hidden_channels, hidden_channels, act=get_activation(act))
                for _ in range(num_blocks)
            ]
        )
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


@register()
class HybridEncoder(nn.Module):
    __share__ = [
        "eval_spatial_size",
    ]

    def __init__(
        self,
        num_feature_levels=5,
        in_channels=[512, 1024, 2048],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        use_encoder_idx=[0, 1, 2, 3, 4],
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
        eval_spatial_size=None,
        use_hybrid=True,
        use_deformable=True,
        enc_n_points=4,
        use_defe=False,
        defe_type="default",
        use_mwas=False,
        mwas_window_size=20
    ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.pos_embeds = []
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        self.use_hybrid = use_hybrid
        self.use_deformable = use_deformable
        self.enc_n_points = enc_n_points
        self.use_defe = use_defe
        self.defe_type = defe_type
        self.use_mwas = use_mwas
        self.mwas_window_size = mwas_window_size

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            proj = nn.Sequential(
                OrderedDict(
                    [
                        ("conv", nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                        ("norm", nn.BatchNorm2d(hidden_dim)),
                    ]
                )
            )

            self.input_proj.append(proj)

        # encoder transformer
        if self.use_deformable:
            if self.num_encoder_layers > 0:
                encoder_layer = DeformableTransformerEncoderLayer(hidden_dim, dim_feedforward,
                                                          dropout, enc_act, num_feature_levels, nhead, enc_n_points)
                self.encoder = DeformableTransformerEncoder(
                    encoder_layer, num_encoder_layers,
                    None, d_model=hidden_dim, 
                    enc_layer_share=False,
                )
                self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, hidden_dim))
            else:
                self.level_embed = None
            
        else:
            if self.num_encoder_layers > 0:
                encoder_layer = TransformerEncoderLayer(
                    hidden_dim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=enc_act,
                )
                self.encoder = nn.ModuleList(
                    [
                        TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
                        for _ in range(len(use_encoder_idx))
                    ]
                )

        if self.use_hybrid:
            # top-down fpn
            self.lateral_convs = nn.ModuleList()
            self.fpn_blocks = nn.ModuleList()
            for _ in range(len(in_channels) - 1, 0, -1):
                self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1))
                self.fpn_blocks.append(
                    RepNCSPELAN4(
                        hidden_dim * 2,
                        hidden_dim,
                        hidden_dim * 2,
                        round(expansion * hidden_dim // 2),
                        round(3 * depth_mult),
                    )
                    # CSPLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, bottletype=VGGBlock)
                )
            # # bottom-up pan
            self.downsample_convs = nn.ModuleList()
            self.pan_blocks = nn.ModuleList()
            for _ in range(len(in_channels) - 1):
                self.downsample_convs.append(
                    nn.Sequential(
                        SCDown(hidden_dim, hidden_dim, 3, 2),
                    )
                )
                self.pan_blocks.append(
                    RepNCSPELAN4(
                        hidden_dim * 2,
                        hidden_dim,
                        hidden_dim * 2,
                        round(expansion * hidden_dim // 2),
                        round(3 * depth_mult),
                    )
                    # CSPLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, bottletype=VGGBlock)
                )

        # DeFE Module
        if self.use_defe:
            if self.use_mwas:
                self.mwas_processor = WindowProcessor(embed_dim=self.hidden_dim, dim_feedforward=dim_feedforward, num_layers=1)
                pass
            if self.defe_type == "light":
                self.DeFE = LiteDeFE()
            else:
                raise ValueError(f"Invalid defe_type: {self.defe_type}")

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            if self.use_deformable:
                if self.num_encoder_layers > 0:
                    nn.init.normal_(self.level_embed)
                for idx, stride in enumerate(self.feat_strides):
                    stride = self.feat_strides[idx]
                    self.pos_embeds.append(self.build_2d_sincos_position_embedding(
                        ceil(self.eval_spatial_size[1] / stride),
                        ceil(self.eval_spatial_size[0] / stride),
                        self.hidden_dim,
                        self.pe_temperature,
                    ))
            else:
                for idx in self.use_encoder_idx:
                    stride = self.feat_strides[idx]
                    self.pos_embeds.append(self.build_2d_sincos_position_embedding(
                        ceil(self.eval_spatial_size[1] / stride),
                        ceil(self.eval_spatial_size[0] / stride),
                        self.hidden_dim,
                        self.pe_temperature,
                    ))

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """ """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        assert (
            embed_dim % 4 == 0
        ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def restore_features(self, src_flatten, spatial_shapes):
        """将展平特征恢复为层级结构"""
        start_idx = 0
        restored_feats = []
        for (H, W) in spatial_shapes:
            end_idx = start_idx + H * W
            feat = src_flatten[:, start_idx:end_idx, :]  # [N, H*W, C]
            N, C = feat.shape[0], feat.shape[2]
            feat = feat.transpose(1, 2).view(N, C, H, W)
            restored_feats.append(feat)
            start_idx = end_idx
        return restored_feats
    

    def adaptive_defe_filter(self, defe_feature, init_thresh=0.05, step=0.01):
        """
        自适应调整阈值直到每个样本找到有效区域
        Args:
            defe_feature: 置信度特征图 [B, 1, H, W]
            init_thresh: 初始阈值
            step: 阈值调整步长
        Returns:
            defe_feature_filtered: 调整后的二值掩码 [B, 1, H, W]
        """
        B = defe_feature.shape[0]
        device = defe_feature.device
        final_mask = torch.zeros_like(defe_feature, dtype=torch.bool)
        
        # 对每个样本独立处理
        for b in range(B):
            # 提取单样本置信图 [1, H, W]
            single_feat = defe_feature[b:b+1]
            current_thresh = init_thresh
            found = False
            
            # 阈值搜索循环
            while current_thresh >= 0:
                mask = (single_feat > current_thresh)
                if mask.any():
                    final_mask[b:b+1] = mask
                    found = True
                    break
                current_thresh = round(current_thresh - step, 2)
            
            # 未找到有效区域则随机选择一个点加强
            if not found:
                final_mask[b:b+1] = torch.zeros_like(single_feat, dtype=torch.bool)
                final_mask[b:b+1][:, random.randint(0, single_feat.shape[1] - 1), random.randint(0, single_feat.shape[2] - 1)] = True
                print(f"Batch {b}: No valid region found, use random point enhancement")
        
        return final_mask


    def forward(self, feats, img_inputs, targets=None):
        out = {"img_inputs": img_inputs}
        
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        if SAVE_INTERMEDIATE_VISUALIZE_RESULT:
            from tools.visualize_src_flatten import visualize_src_flatten
            visualize_src_flatten(src_flatten=proj_feats[0].permute(0, 2, 3, 1), spatial_shapes=[proj_feats[0].shape[2:]], savename="backbone_output_0", is_flatten=False)

        # DeFE Module
        if self.use_defe:
            defe_feature, reg_value = self.DeFE(proj_feats[0])
            W, H = proj_feats[1].shape[2:]
            out["defe"] = {"reg_value": reg_value, "density_map": defe_feature}
            defe_feature_pooled = F.adaptive_max_pool2d(defe_feature, (H // self.mwas_window_size, W // self.mwas_window_size))
            out["defe"]["defe_feature"] = defe_feature
            out["defe"]["density_map_pooled"] = defe_feature_pooled
            if self.use_mwas:
                W, H = proj_feats[1].shape[2:]
                defe_feature_filtered = self.adaptive_defe_filter(F.interpolate(defe_feature_pooled, size=(H, W), mode="bilinear", align_corners=True)).float()
                glob_pos_embed = self.build_2d_sincos_position_embedding(W, H, embed_dim=self.hidden_dim).permute(0, 2, 1).view(-1, H, W).to(proj_feats[1].device)
                enhanced_memory, defe_window_mask = self.mwas_processor(proj_feats[1], defe_feature_filtered, self.mwas_window_size, glob_pos_embed)
                proj_feats[1] = enhanced_memory
                out["defe"]["defe_window_mask"] = defe_window_mask
                if SAVE_INTERMEDIATE_VISUALIZE_RESULT:
                    from tools.visualize_src_flatten import visualize_src_flatten
                    visualize_src_flatten(src_flatten=proj_feats[0].permute(0, 2, 3, 1), spatial_shapes=[proj_feats[0].shape[2:]], savename="encoder_output_0", is_flatten=False)
            if SAVE_INTERMEDIATE_VISUALIZE_RESULT:
                from tools.visualize_src_flatten import visualize_src_flatten
                visualize_src_flatten(defe_feature.permute(0, 2, 3, 1), [(defe_feature.shape[2:4])], "defe_feature", False)
                visualize_src_flatten(defe_feature_pooled.permute(0, 2, 3, 1), [(defe_feature_pooled.shape[2:4])], "defe_feature_pooled", False)
                visualize_src_flatten(defe_feature_filtered.permute(0, 2, 3, 1), [(defe_feature_filtered.shape[2:4])], "defe_feature_filtered", False)
            out["defe"]["gt_density_map"] = []
            if targets is not None:
                B, C, H, W = img_inputs.shape
                heatmap_generator = GaussHeatmapGenerator(img_size=(H, W))
                for b in range(B):
                    boxes = targets[b]["boxes"]
                    # conver xyxy to center_xywh
                    if not self.training:
                        boxes[:, 0] = boxes[:, 0] / W  # x1
                        boxes[:, 1] = boxes[:, 1] / H  # y1
                        boxes[:, 2] = boxes[:, 2] / W  # x2
                        boxes[:, 3] = boxes[:, 3] / H  # y2
                        x1y1 = boxes[:, :2]
                        x2y2 = boxes[:, 2:]
                        cxcy = (x1y1 + x2y2) / 2
                        wh = x2y2 - x1y1
                        boxes = torch.cat([cxcy, wh], dim=1)
                    heatmap = heatmap_generator(boxes)
                    out["defe"]["gt_density_map"].append(heatmap)
                out["defe"]["gt_density_map"] = torch.stack(out["defe"]["gt_density_map"]).to(img_inputs.device)
                if SAVE_INTERMEDIATE_VISUALIZE_RESULT:
                    from tools.visualize_src_flatten import visualize_src_flatten
                    visualize_src_flatten( out["defe"]["gt_density_map"].permute(0, 2, 3, 1), [(heatmap.shape[2:4])], "heatmap_gt", False)

        # encoder
        if self.use_deformable:
            if self.num_encoder_layers > 0:
                src_flatten = []
                mask_flatten = []
                lvl_pos_embed_flatten = []
                spatial_shapes = []
                masks = []
                for lvl, src in enumerate(proj_feats):
                    bs, c, h, w = src.shape
                    spatial_shape = (h, w)
                    spatial_shapes.append(spatial_shape)

                    # generate mask and pos_embed
                    if self.training or self.eval_spatial_size is None:
                        pos_embed = self.build_2d_sincos_position_embedding(
                            w, h, self.hidden_dim, self.pe_temperature
                        ).to(src.device)
                    else:
                        pos_embed = self.pos_embeds[lvl].to(src.device)

                    # generate all False mask which shape is （bs, hw)
                    mask = torch.zeros((bs, h, w), dtype=torch.bool, device=src.device)
                    masks.append(mask.clone())
                    mask = mask.flatten(1)
                    src = src.flatten(2).transpose(1, 2) # bs, hw, c
                    if self.num_feature_levels > 1 and self.level_embed is not None:
                        lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
                    else:
                        lvl_pos_embed = pos_embed
                    lvl_pos_embed_flatten.append(lvl_pos_embed)
                    src_flatten.append(src)
                    mask_flatten.append(mask)
                
                src_flatten = torch.cat(src_flatten, 1)    # bs, \sum{hxw}, c 
                mask_flatten = torch.cat(mask_flatten, 1)   # bs, \sum{hxw}
                lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) # bs, \sum{hxw}, c 
                spatial_shapes_tensor = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
                level_start_index = torch.cat((spatial_shapes_tensor.new_zeros((1, )), spatial_shapes_tensor.prod(1).cumsum(0)[:-1]))
                valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

                if SAVE_INTERMEDIATE_VISUALIZE_RESULT:
                    from tools.visualize_src_flatten import visualize_src_flatten
                    visualize_src_flatten(src_flatten=src_flatten, spatial_shapes=spatial_shapes, savename="encoder_input")

                memory, enc_intermediate_output = self.encoder(
                    src_flatten, 
                    pos=lvl_pos_embed_flatten,
                    spatial_shapes=spatial_shapes_tensor,
                    level_start_index=level_start_index,
                    valid_ratios=valid_ratios,
                    key_padding_mask=mask_flatten,
                )

                proj_feats = self.restore_features(memory, spatial_shapes)

                if SAVE_INTERMEDIATE_VISUALIZE_RESULT:
                    from tools.visualize_src_flatten import visualize_src_flatten
                    visualize_src_flatten(src_flatten=memory, spatial_shapes=spatial_shapes, savename="encoder_output")
        else:
            if self.num_encoder_layers > 0:
                for i, enc_ind in enumerate(self.use_encoder_idx):
                    h, w = proj_feats[enc_ind].shape[2:]
                    # flatten [B, C, H, W] to [B, HxW, C]
                    src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                    if self.training or self.eval_spatial_size is None:
                        pos_embed = self.build_2d_sincos_position_embedding(
                            w, h, self.hidden_dim, self.pe_temperature
                        ).to(src_flatten.device)
                    else:
                        pos_embed = self.pos_embeds[i].to(src_flatten.device)
                    memory: torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                    proj_feats[enc_ind] = (
                        memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                    )
        
        if self.use_hybrid:
            # broadcasting and fusion
            inner_outs = [proj_feats[-1]]
            for idx in range(len(self.in_channels) - 1, 0, -1):
                feat_heigh = inner_outs[0]
                feat_low = proj_feats[idx - 1]
                feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
                inner_outs[0] = feat_heigh
                upsample_feat = F.interpolate(feat_heigh, size=(feat_low.shape[2], feat_low.shape[3]), mode="bilinear", align_corners=True)
                inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                    torch.concat([upsample_feat, feat_low], dim=1)
                )
                inner_outs.insert(0, inner_out)

            outs = [inner_outs[0]]
            for idx in range(len(self.in_channels) - 1):
                feat_low = outs[-1]
                feat_height = inner_outs[idx + 1]
                downsample_feat = self.downsample_convs[idx](feat_low)
                pan_out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
                outs.append(pan_out)
        else:
            outs = proj_feats

        out["feats"] = outs

        return out
