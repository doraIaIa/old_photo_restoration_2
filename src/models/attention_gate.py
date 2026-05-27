from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    def __init__(self, gating_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gating_projection = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.skip_projection = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.attention = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if g.ndim != 4 or x.ndim != 4:
            raise ValueError("AttentionGate yêu cầu tensor 4 chiều [B, C, H, W]")

        gating = self.gating_projection(g)
        if gating.shape[-2:] != x.shape[-2:]:
            gating = F.interpolate(gating, size=x.shape[-2:], mode="bilinear", align_corners=False)

        skip = self.skip_projection(x)
        attention_map = self.attention(gating + skip)
        return x * attention_map
