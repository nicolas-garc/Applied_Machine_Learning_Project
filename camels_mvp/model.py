from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular"),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular"),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class ConditionedUNet(nn.Module):
    def __init__(self, param_dim: int = 6, base_channels: int = 16) -> None:
        super().__init__()
        self.enc1 = ConvBlock(1, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)
        self.param_encoder = nn.Sequential(
            nn.Linear(param_dim, base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Linear(base_channels * 8, base_channels * 8),
        )

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.output_layer = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, inputs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        skip1 = self.enc1(inputs)
        skip2 = self.enc2(self.pool(skip1))
        skip3 = self.enc3(self.pool(skip2))

        bottleneck = self.bottleneck(self.pool(skip3))
        param_bias = self.param_encoder(params).unsqueeze(-1).unsqueeze(-1)
        bottleneck = bottleneck + param_bias

        up3 = self.up3(bottleneck)
        dec3 = self.dec3(torch.cat([up3, skip3], dim=1))
        up2 = self.up2(dec3)
        dec2 = self.dec2(torch.cat([up2, skip2], dim=1))
        up1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([up1, skip1], dim=1))
        return self.output_layer(dec1)

