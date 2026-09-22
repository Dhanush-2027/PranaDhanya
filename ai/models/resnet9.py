"""
ResNet9 Architecture with Residual Connections
======================================================
Custom lightweight 9-layer ResNet CNN for image classification.
Follows the canonical architecture:
    ConvBlock -> ConvBlock + Residual -> ConvBlock -> ConvBlock + Residual -> Classifier

Parametric Layers:
1. Prep Conv: Conv2d(3, c)
2. Layer 1:   Conv2d(c, 2*c) + MaxPool2d(2)
3. ResBlock1: Conv2d(2*c, 2*c)
4. ResBlock1: Conv2d(2*c, 2*c)
5. Layer 2:   Conv2d(2*c, 4*c) + MaxPool2d(2)
6. Layer 3:   Conv2d(4*c, 8*c) + MaxPool2d(2)
7. ResBlock2: Conv2d(8*c, 8*c)
8. ResBlock2: Conv2d(8*c, 8*c)
9. Linear:    Linear(8*c, num_classes)
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


class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convolutions, batch normalization, and skip connection."""
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
        out = self.relu(out + residual)
        return out


# Alias for backward compatibility
BasicBlock = ResidualBlock


class ResNet9(nn.Module):
    """
    ResNet9 - Canonical 9-layer Deep Residual Neural Network.
    
    Structure:
        Input (3xHxW, e.g. 3x64x64, 3x112x112, 3x224x224)
            ↓
        Prep Conv (3 → c)
            ↓
        Layer 1 Conv (c → 2c, MaxPool /2)
            ↓
        ResBlock 1 (2c → 2c, 2 Convs + Skip)
            ↓
        Layer 2 Conv (2c → 4c, MaxPool /2)
            ↓
        Layer 3 Conv (4c → 8c, MaxPool /2)
            ↓
        ResBlock 2 (8c → 8c, 2 Convs + Skip)
            ↓
        Global Average Pooling (AdaptiveAvgPool2d(1, 1))
            ↓
        Dropout(0.2) + Linear Classifier (8c → num_classes)
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 10, base_channels: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        c = base_channels
        self.base_channels = c
        
        # 1. Prep layer (Layer 1)
        self.prep = conv_block(in_channels, c, pool=False)
        
        # 2. Stage 1: Conv(c->2c, pool) (Layer 2) + Residual(2c) (Layers 3 & 4)
        self.layer1 = conv_block(c, c * 2, pool=True)
        self.res1 = ResidualBlock(c * 2, dropout=dropout)
        
        # 3. Stage 2: Conv(2c->4c, pool) (Layer 5)
        self.layer2 = conv_block(c * 2, c * 4, pool=True)
        
        # 4. Stage 3: Conv(4c->8c, pool) (Layer 6) + Residual(8c) (Layers 7 & 8)
        self.layer3 = conv_block(c * 4, c * 8, pool=True)
        self.res2 = ResidualBlock(c * 8, dropout=dropout)
        
        # 5. Classifier Head (Layer 9)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(c * 8, num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.prep(x)
        out = self.layer1(out)
        out = self.res1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.res2(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.classifier(out)
        return out


def count_parameters(model: nn.Module) -> int:
    """Returns the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
