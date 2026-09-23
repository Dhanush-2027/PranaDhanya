"""
ResNet9 Architecture with Adaptive Support for 32/64-channel & Sequential/Named Residual Blocks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def conv_block(in_channels: int, out_channels: int, pool: bool = False, pool_size: int = 2) -> nn.Sequential:
    """Standard Conv2d -> BatchNorm2d -> ReLU block with optional MaxPool2d."""
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(pool_size))
    return nn.Sequential(*layers)


class LegacyResidualBlock(nn.Module):
    """Residual block with named conv1/conv2 sub-layers for legacy weights."""
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        return self.relu(out + residual)


class ResNet9(nn.Module):
    """
    ResNet9 Architecture supporting both standard 64-channel and legacy 32-channel weights.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 11,
        base_channels: int = 64,
        dropout: float = 0.2,
        legacy_mode: bool = False
    ) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        c = base_channels
        self.base_channels = c
        self.legacy_mode = legacy_mode
        
        # 1. Prep layer
        self.prep = conv_block(in_channels, c, pool=False)
        
        # 2. Stage 1: Conv(c->2c, pool) + Residual(2c)
        self.layer1 = conv_block(c, c * 2, pool=True)
        if legacy_mode:
            self.res1 = LegacyResidualBlock(c * 2, dropout=dropout)
        else:
            self.res1 = nn.Sequential(
                conv_block(c * 2, c * 2),
                conv_block(c * 2, c * 2)
            )
        
        # 3. Stage 2: Conv(2c->4c, pool)
        self.layer2 = conv_block(c * 2, c * 4, pool=True)
        
        # 4. Stage 3: Conv(4c->8c, pool) + Residual(8c)
        self.layer3 = conv_block(c * 4, c * 8, pool=True)
        if legacy_mode:
            self.res2 = LegacyResidualBlock(c * 8, dropout=dropout)
        else:
            self.res2 = nn.Sequential(
                conv_block(c * 8, c * 8),
                conv_block(c * 8, c * 8)
            )
        
        # 5. Classifier Head
        if legacy_mode:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(c * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.prep(x)
        out = self.layer1(out)
        if self.legacy_mode:
            out = self.res1(out)
        else:
            out = out + self.res1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.legacy_mode:
            out = self.res2(out)
        else:
            out = out + self.res2(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.classifier(out)
        return out


def build_resnet9_from_state_dict(state_dict, num_classes):
    """
    Intelligently inspects state_dict keys and dimensions to instantiate the exact matching ResNet9 variant.
    """
    # Detect base channels from prep layer
    prep_weight = state_dict.get('prep.0.weight', None)
    base_channels = prep_weight.shape[0] if prep_weight is not None else 64
    
    # Detect legacy vs modern block naming
    legacy_mode = any('res1.conv1' in k for k in state_dict.keys())
    
    model = ResNet9(
        in_channels=3,
        num_classes=num_classes,
        base_channels=base_channels,
        legacy_mode=legacy_mode
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model
