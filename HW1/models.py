#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model definitions for image classification.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNeXt101_32X8D_Weights


class GeM(nn.Module):
    """Generalized mean pooling."""

    def __init__(self, p=3.0, eps=1e-6, trainable=True):
        super().__init__()
        if trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = float(p)
        self.eps = eps

    def forward(self, x):
        p = self.p if isinstance(self.p, torch.Tensor) else torch.tensor(
            self.p, device=x.device, dtype=x.dtype
        )
        x = x.clamp(min=self.eps)
        x = x.pow(p)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.pow(1.0 / p)
        return x

    def __repr__(self):
        if isinstance(self.p, torch.Tensor):
            p_val = self.p.data.tolist()[0]
        else:
            p_val = self.p
        return f"GeM(p={p_val:.4f}, eps={self.eps})"


class EnhancedResNeXt101(nn.Module):
    """
    ResNeXt101 with GeM pooling and channel attention.
    """

    def __init__(self, num_classes=100, dropout_prob=0.5):
        """
        Initialize the model.
        """
        super(EnhancedResNeXt101, self).__init__()

        # Load pretrained backbone
        base_model = models.resnext101_32x8d(
            weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2)

        # Print backbone size
        total_params = sum(p.numel() for p in base_model.parameters())
        print(f"Base ResNeXt101_32X8D parameters: {total_params:,} "
              f"({total_params/1e6:.2f}M)")

        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(base_model.children())[:-2])

        # Pooling layer
        self.pool = GeM(p=3.0, trainable=True)

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.Linear(2048, 2048 // 16),
            nn.ReLU(inplace=True),
            nn.Linear(2048 // 16, 2048),
            nn.Sigmoid()
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_prob),
            nn.Linear(2048, num_classes)
        )

        # Print total model size
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Enhanced ResNeXt101_32X8D parameters: {total_params:,} "
              f"({total_params/1e6:.2f}M)")

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize added layers."""
        for m in self.channel_attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass.
        """
        # Extract features
        x = self.features(x)

        # Pool features
        x_pool = self.pool(x).view(x.size(0), -1)

        # Apply channel attention
        att = self.channel_attention(x_pool)
        x_att = x_pool * att

        # Predict class scores
        out = self.classifier(x_att)

        return out


def create_model(model_type="resnext101", num_classes=100, dropout_prob=0.5):
    """
    Create a model instance.
    """
    if model_type == "resnext101":
        return EnhancedResNeXt101(
            num_classes=num_classes,
            dropout_prob=dropout_prob
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
