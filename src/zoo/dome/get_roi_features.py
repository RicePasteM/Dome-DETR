'''
Dome-DETR: Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
'''

import torch
import torch.nn.functional as F
from skimage.measure import label, regionprops
from tools.visualize_src_flatten import visualize_src_flatten
import numpy as np
from .utils import get_activation
import torch.nn as nn   
import os
import copy

SAVE_INTERMEDIATE_VISUALIZE_RESULT = os.environ.get('SAVE_INTERMEDIATE_VISUALIZE_RESULT', 'False') == 'True'
print(f"SAVE_INTERMEDIATE_VISUALIZE_RESULT: {SAVE_INTERMEDIATE_VISUALIZE_RESULT}")


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output
    

# transformer
class AxisPermutedEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    def forward(self, q, k, v, src_mask=None, key_padding_mask=None) -> torch.Tensor:
        src = residual = v
        if self.normalize_before:
            src = self.norm1(src)

        src, _ = self.self_attn(
            q,
            k,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=key_padding_mask,
        )

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src
    

class AxisPermutedEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(AxisPermutedEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None, glob_pos_embeds=None) -> torch.Tensor:
        output = src
        B, C, _, _ = glob_pos_embeds.shape
        glob_pos_embeds = glob_pos_embeds.reshape(B, C, -1).permute(0, 2, 1)
        pos_embed = pos_embed.reshape(C, -1).permute(1, 0).unsqueeze(0)
        for layer in self.layers:
            q = k = self.with_pos_embed(output, glob_pos_embeds + pos_embed)
            output = layer(q, k, output, src_mask=src_mask)

            output = output.permute(1, 0, 2).contiguous()
            q = k = self.with_pos_embed(output, (glob_pos_embeds + pos_embed).permute(1, 0, 2).contiguous())
            output = layer(q, k, output, src_mask=src_mask)
            output = output.permute(1, 0, 2).contiguous()

        if self.norm is not None:
            output = self.norm(output)

        return output
        


class WindowProcessor(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, dim_feedforward=1024, num_layers=1, dropout=0.0, activation="relu"):
        """
        窗口处理类
        Args:
            embed_dim: 位置编码维度
            use_residual: 是否使用残差连接
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # 相对位置编码器
        self.rel_pos_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )

        encoder_layer = AxisPermutedEncoderLayer(
            self.embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

        self.window_encoder = AxisPermutedEncoder(encoder_layer, self.num_layers)

    @staticmethod
    def _should_use_export_path():
        return torch.onnx.is_in_onnx_export()

    def forward(self, backbone_memory, defe_feature_filtered, window_size, glob_pos_embed, mask_stride=None):
        if self._should_use_export_path():
            return self._forward_export(
                backbone_memory,
                defe_feature_filtered,
                window_size,
                glob_pos_embed,
                mask_stride,
            )
        return self._forward_default(
            backbone_memory,
            defe_feature_filtered,
            window_size,
            glob_pos_embed,
        )

    def _forward_default(self, backbone_memory, defe_feature_filtered, window_size, glob_pos_embed):
        """
        前向传播
        Args:
            backbone_memory: 原特征图 [B, C, H, W]
            defe_feature_filtered: 二值图 [B, 1, H, W] 
            window_size: 窗口大小 (像素/特征点数)
            feature_enhancer: 可选的特征增强模块
        """

        B, C, H, W = backbone_memory.shape

        assert H % window_size == 0 and W % window_size == 0, "H and W must be divisible by window_size"
        
        num_win_h = H // window_size
        num_win_w = W // window_size

        rel_pos_embed = self._get_rel_embedding((window_size, window_size)).to(backbone_memory.device)

        reconstructed = backbone_memory.clone()
        windows, defe_mask = self._prepare_windows(backbone_memory, defe_feature_filtered, window_size)

        if SAVE_INTERMEDIATE_VISUALIZE_RESULT:
            window_mask_vis = defe_mask.float().unsqueeze(-1)  # [B, num_win_h, num_win_w, 1]
            visualize_src_flatten(window_mask_vis, [(num_win_h, num_win_w)], "defe_window_mask", False)

        for b in range(B):
            valid_windows = torch.nonzero(defe_mask[b])
            if len(valid_windows) == 0:
                with open("no_valid_windows.log", "a") as f:
                    f.write(f"batch {b} has no valid windows\n")
                    f.write(f"defe_feature_filtered_nonzero: {torch.nonzero(defe_feature_filtered[b])}\n")
                    f.write(f"defe_feature_filtered: {defe_feature_filtered[b].size()}\n")
                    f.write(f"defe_mask: {defe_mask[b].size()}\n")
            assert len(valid_windows) > 0, "No valid windows found"

            window_features, glob_pos_embeds = self._process_windows(
                windows[b],
                valid_windows,
                H,
                W,
                window_size,
                glob_pos_embed,
            )

            encoded_features = self._encode_features(window_features, rel_pos_embed, glob_pos_embeds)

            self._reconstruct_features(
                reconstructed,
                b,
                encoded_features,
                valid_windows,
                window_size,
                window_size,
            )

        return reconstructed, defe_mask

    def _forward_export(self, backbone_memory, defe_feature_filtered, window_size, glob_pos_embed, mask_stride):
        """
        导出阶段使用向量化实现，避免窗口级 Python 循环进入 ONNX 图。
        """
        B, C, H, W = backbone_memory.shape

        assert H % window_size == 0 and W % window_size == 0, "H and W must be divisible by window_size"

        num_win_h = H // window_size
        num_win_w = W // window_size

        rel_pos_embed = self._get_rel_embedding((window_size, window_size)).to(backbone_memory.device)

        windows, defe_mask = self._prepare_windows_export(
            backbone_memory,
            defe_feature_filtered,
            window_size,
        )

        if SAVE_INTERMEDIATE_VISUALIZE_RESULT:
            window_mask_vis = defe_mask.float().unsqueeze(-1)
            visualize_src_flatten(window_mask_vis, [(num_win_h, num_win_w)], "defe_window_mask", False)

        glob_pos_windows = (
            glob_pos_embed.view(C, num_win_h, window_size, num_win_w, window_size)
            .permute(1, 3, 0, 2, 4)
            .unsqueeze(0)
            .expand(B, -1, -1, -1, -1, -1)
        )

        num_windows = num_win_h * num_win_w
        token_len = window_size * window_size

        features = windows.reshape(B, num_windows, C, window_size, window_size)
        features = features.reshape(B, num_windows, C, token_len).permute(0, 1, 3, 2)
        glob_pos_embeds = glob_pos_windows.reshape(B, num_windows, C, window_size, window_size)
        glob_pos_embeds = glob_pos_embeds.reshape(B, num_windows, C, token_len).permute(0, 1, 3, 2)
        rel_pos = rel_pos_embed.reshape(C, token_len).permute(1, 0).view(1, 1, token_len, C)
        valid_mask = defe_mask.reshape(B, num_windows)

        for layer in self.window_encoder.layers:
            within_input = features.reshape(B * num_windows, token_len, C)
            within_pos = glob_pos_embeds.reshape(B * num_windows, token_len, C) + rel_pos.reshape(
                1, token_len, C
            )
            within_out = layer(
                within_input + within_pos,
                within_input + within_pos,
                within_input,
            )
            features = within_out.reshape(B, num_windows, token_len, C)

            cross_input = features.permute(0, 2, 1, 3).reshape(B * token_len, num_windows, C)
            cross_pos = (glob_pos_embeds + rel_pos).permute(0, 2, 1, 3).reshape(
                B * token_len, num_windows, C
            )
            key_padding_mask = (~valid_mask).unsqueeze(1).expand(B, token_len, num_windows)
            key_padding_mask = key_padding_mask.reshape(B * token_len, num_windows)
            cross_out = layer(
                cross_input + cross_pos,
                cross_input + cross_pos,
                cross_input,
                key_padding_mask=key_padding_mask,
            )
            features = cross_out.reshape(B, token_len, num_windows, C).permute(0, 2, 1, 3)

        if self.window_encoder.norm is not None:
            features = self.window_encoder.norm(features)

        encoded_features = features.permute(0, 1, 3, 2).reshape(
            B, num_windows, C, window_size, window_size
        )
        encoded_features = encoded_features * valid_mask.view(B, num_windows, 1, 1, 1).to(
            encoded_features.dtype
        )
        encoded_features = encoded_features.reshape(B, num_win_h, num_win_w, C, window_size, window_size)
        updates = encoded_features.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        return backbone_memory + updates, defe_mask

    def _prepare_windows_export(self, features, mask, window_size):
        """
        导出阶段使用固定窗口池化，避免动态 kernel_size 进入 ONNX。
        """
        B, C, H_feat, W_feat = features.shape

        num_win_h = H_feat // window_size
        num_win_w = W_feat // window_size

        windows = features.view(B, C, num_win_h, window_size, num_win_w, window_size).permute(0, 2, 4, 1, 3, 5)

        pooled_mask = F.max_pool2d(
            mask.float(),
            kernel_size=(window_size, window_size),
            stride=(window_size, window_size),
        )
        defe_mask = pooled_mask.squeeze(1) > 0

        return windows, defe_mask


    def _prepare_windows(self, features, mask, window_size, mask_stride=None):
        """预处理窗口划分和掩码（最大池化优化版）"""
        B, C, H_feat, W_feat = features.shape
        H_mask, W_mask = mask.shape[-2:]

        num_win_h = H_feat // window_size
        num_win_w = W_feat // window_size

        kernel_h = H_mask // H_feat * window_size
        kernel_w = W_mask // W_feat * window_size
        stride_h = kernel_h
        stride_w = kernel_w

        # 划分特征图窗口 [B, num_win_h, num_win_w, C, window_size, window_size]
        windows = features.view(B, C, num_win_h, window_size, num_win_w, window_size).permute(0, 2, 4, 1, 3, 5)

        # 最大池化统计窗口有效性
        mask_float = mask.float()  # 转换为浮点型以支持池化
        pooled_mask = F.max_pool2d(
            mask_float,
            kernel_size=(kernel_h, kernel_w),
            stride=(stride_h, stride_w)
        )
        defe_mask = (pooled_mask.squeeze(1) > 0)  # [B, num_win_h, num_win_w]

        return windows, defe_mask


    def _process_windows(self, windows, valid_indices, H, W, window_size, glob_pos_embed):
        """处理窗口特征并添加位置编码"""
        batch_features = []
        batch_glob_pos_embed = []
        
        # Grid dimensions based on window_size
        num_win_h = H // window_size
        num_win_w = W // window_size
        
        for i, j in valid_indices:
            # 原始窗口特征 [C, h, w]
            win_feat = windows[i, j]
            
            # 拼接特征和位置编码
            combined = win_feat
            
            batch_features.append(combined.unsqueeze(0))
            batch_glob_pos_embed.append(self._get_abs_embedding(glob_pos_embed, i, j, window_size, window_size))
        
        return torch.cat(batch_features, dim=0), torch.stack(batch_glob_pos_embed)  # [N, C]

    
    def _encode_features(self, features, rel_pos_embed, glob_pos_embeds):
        """三层编码处理"""
        # 调整维度 [N, C+D, h, w] -> [N, L, C+D]
        B, C, h, w = features.shape
        features = features.view(B, C, -1).permute(0, 2, 1)

        features = self.window_encoder(features, pos_embed=rel_pos_embed, glob_pos_embeds=glob_pos_embeds)
            
        # 恢复空间维度
        return features.permute(0, 2, 1).view(B, C, h, w)

    def _reconstruct_features(self, reconstructed, batch_idx, feats, indices, win_h, win_w):
        """特征图重建"""
        for idx, (i, j) in enumerate(indices):
            h_start = i * win_h
            h_end = h_start + win_h
            w_start = j * win_w
            w_end = w_start + win_w
            
            reconstructed[batch_idx, :, h_start:h_end, w_start:w_end] += feats[idx]

    def _get_abs_embedding(self, global_emb, i, j, win_h, win_w):
        # global_emb形状: [C, H, W]
        x0 = j * win_w
        y0 = i * win_h
        return global_emb[:, y0:y0+win_h, x0:x0+win_w]  # [256, win_h, win_w]

    def _get_rel_embedding(self, window_size):
        """相对位置编码"""
        h, w = window_size
        coords = self._get_relative_coords(h, w)
        return self.rel_pos_encoder(coords.to(self.rel_pos_encoder[0].weight.device)).permute(2,0,1)

    @staticmethod
    def _get_relative_coords(h, w):
        """生成归一化坐标矩阵"""
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))
        return torch.stack([grid_x/(w-1), grid_y/(h-1)], dim=-1)
