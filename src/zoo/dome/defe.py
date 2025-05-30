'''
Dome-DETR: Dome-DETR: DETR with Density-Oriented Feature-Query Manipulation for Efficient Tiny Object Detection
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LightweightAttention(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        att = self.gap(x).view(b, c)
        att = self.fc(att).view(b, c, 1, 1)
        return x * att.expand_as(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        assert isinstance(in_ch, int), "Input channels must be integer"
        
        self.depthwise = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=in_ch
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(x)

class OptimizedDeFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = [
            (256, 1),
            (256, 2),
            (256, 3),
            (256, 1),
            (256, 1)
        ]
        
        layers = []
        in_ch = 256
        
        for idx, (out_ch, dilation) in enumerate(self.cfg):
            layers += [
                DepthwiseSeparableConv(in_ch, out_ch, dilation),
                nn.BatchNorm2d(out_ch)
            ]
            in_ch = out_ch
            
            if idx == 2:
                layers.append(LightweightAttention(out_ch))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class LiteDeFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.AvgPool2d(kernel_size=2)
        )
        
        self.defe = OptimizedDeFE()
        
        self.density_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 1, 1),  # 输出 [B,1,H,W]
            nn.Sigmoid()
        )

        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        x = self.conv1(features)
        
        x = self.defe(x)
    
        density = F.interpolate(
            self.density_head(x),  # 用x生成密度图
            scale_factor=2,
            mode='bilinear',
            align_corners=False
        )

        # 对density进行0-1归一化
        if density.max() > 0:
            density = density / density.max()

        reg_value = self.regression_head(x)
        
        return density, reg_value
    

class GaussHeatmapGenerator:
    def __init__(self, img_size=(640, 640), sigma_ratio=1.2):
        self.img_size = img_size
        self.sigma_ratio = sigma_ratio

    def __call__(self, bboxes):
        H, W = self.img_size
        heatmap = torch.zeros((H, W), dtype=torch.float32)
        
        for box in bboxes:
            x_center, y_center, width, height = box
            x_center_px = int(x_center * W)
            y_center_px = int(y_center * H)
            w_px = max(int(width * W), 1)
            h_px = max(int(height * H), 1)
            
            sigma_x = max(w_px * self.sigma_ratio, 1.0)
            sigma_y = max(h_px * self.sigma_ratio, 1.0)
            
            kernel = self._gaussian_kernel(sigma_x, sigma_y)
            if kernel.numel() == 0:  # 使用 numel() 替代 size
                continue
            
            k_h, k_w = kernel.shape
            radius_x = k_w // 2
            radius_y = k_h // 2
            
            # 计算粘贴区域
            x_start = max(x_center_px - radius_x, 0)
            y_start = max(y_center_px - radius_y, 0)
            x_end = min(x_center_px + radius_x + 1, W)
            y_end = min(y_center_px + radius_y + 1, H)
            
            # 计算核的裁剪区域
            k_start_x = max(radius_x - (x_center_px - x_start), 0)
            k_start_y = max(radius_y - (y_center_px - y_start), 0)
            k_end_x = k_w - max((x_center_px + radius_x + 1) - x_end, 0)
            k_end_y = k_h - max((y_center_px + radius_y + 1) - y_end, 0)
            
            kernel_cropped = kernel[k_start_y:k_end_y, k_start_x:k_end_x]
            
            # 确保区域有效
            if kernel_cropped.numel() == 0:  # 使用 numel() 替代 size
                continue
                
            # 确保尺寸匹配
            patch_h = y_end - y_start
            patch_w = x_end - x_start
            
            # 确保核的尺寸与目标区域完全匹配
            if kernel_cropped.shape != (patch_h, patch_w):
                kernel_cropped = kernel_cropped[:patch_h, :patch_w]
                
            # 叠加到热图
            heatmap[y_start:y_end, x_start:x_end] += kernel_cropped
        
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        return heatmap.unsqueeze(0)

    def _gaussian_kernel(self, sigma_x, sigma_y):
        sigma_x = max(sigma_x, 0.1)  # 确保不会太小
        sigma_y = max(sigma_y, 0.1)
        kernel_w = int(6 * sigma_x) + 1
        kernel_h = int(6 * sigma_y) + 1
        
        if kernel_w % 2 == 0:
            kernel_w += 1
        if kernel_h % 2 == 0:
            kernel_h += 1
            
        # 使用 torch.arange 替代 np.arange
        x = torch.arange(kernel_w, dtype=torch.float32) - (kernel_w // 2)
        y = torch.arange(kernel_h, dtype=torch.float32) - (kernel_h // 2)
        
        # 使用 torch.meshgrid 替代 np.meshgrid
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # 使用 torch 操作计算高斯核
        kernel = torch.exp(-(xx**2 / (2 * sigma_x**2) + yy**2 / (2 * sigma_y**2)))
        
        # 归一化
        kernel_sum = kernel.sum()
        if kernel_sum > 0:
            kernel = kernel / kernel_sum
            
        return kernel