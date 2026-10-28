"""EfficientNet model architecture for plant disease detection."""

import torch
import torch.nn as nn
import timm


class EfficientNetPlantDisease(nn.Module):
    """
    EfficientNet-B3 model for plant disease classification.
    Uses pre-trained ImageNet weights with custom classifier head.
    """
    
    def __init__(self, num_classes=71, model_name='efficientnet_b3', pretrained=True):
        super(EfficientNetPlantDisease, self).__init__()
        
        # Load pretrained EfficientNet
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove original classifier
                global_pool='avg'  # Global average pooling
            )
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Using randomly initialized model instead.")
            self.backbone = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=0,
                global_pool='avg'
            )
        
        # Get feature dimension
        num_features = self.backbone.num_features
        
        # Custom classifier head with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
        
        # Initialize classifier weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier layers."""
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        """Extract features before classifier (for analysis)."""
        return self.backbone(x)


def build_efficientnet_model(num_classes, model_name='efficientnet_b3', pretrained=True):
    """
    Factory function to create EfficientNet model.
    
    Args:
        num_classes: Number of disease classes
        model_name: EfficientNet variant (b0, b1, b2, b3, b4, etc.)
        pretrained: Whether to use ImageNet pretrained weights
    
    Returns:
        EfficientNetPlantDisease model
    """
    model = EfficientNetPlantDisease(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained
    )
    return model


if __name__ == '__main__':
    # Test model creation
    model = build_efficientnet_model(num_classes=71)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"Model output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
