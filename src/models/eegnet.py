"""Compact EEGNet model for 22-channel motor-imagery EEG trials.

The model accepts tensors shaped ``(batch, 1, channels, samples)`` and
returns one unnormalised logit per class.  It deliberately has no data-loading
or normalization logic; those remain in the repository's data contract.
"""

from __future__ import annotations

import torch
from torch import nn


class EEGNet(nn.Module):
    """EEGNet-style classifier for the agreed 22-channel, 750-sample input."""

    def __init__(
        self,
        channels: int = 22,
        samples: int = 750,
        classes: int = 4,
        temporal_filters: int = 8,
        depth_multiplier: int = 2,
        temporal_kernel: int = 64,
        dropout: float = 0.5,
        normalization: str = "batch",
    ) -> None:
        super().__init__()
        if channels <= 0 or samples <= 0 or classes <= 1:
            raise ValueError("channels and samples must be positive; classes must exceed one")
        if temporal_filters <= 0 or depth_multiplier <= 0:
            raise ValueError("temporal_filters and depth_multiplier must be positive")

        spatial_filters = temporal_filters * depth_multiplier
        if normalization not in {"batch", "group"}:
            raise ValueError("normalization must be 'batch' or 'group'")
        norm = lambda features: nn.BatchNorm2d(features) if normalization == "batch" else nn.GroupNorm(1, features)
        self.channels = channels
        self.samples = samples
        self.classes = classes

        self.features = nn.Sequential(
            nn.Conv2d(1, temporal_filters, (1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False),
            norm(temporal_filters),
            nn.Conv2d(
                temporal_filters,
                spatial_filters,
                (channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            norm(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(spatial_filters, spatial_filters, (1, 16), padding=(0, 8), groups=spatial_filters, bias=False),
            nn.Conv2d(spatial_filters, spatial_filters, (1, 1), bias=False),
            norm(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            feature_shape = self.features(torch.zeros(1, 1, channels, samples)).shape
        self.classifier = nn.Linear(feature_shape.numel(), classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(batch, classes)`` logits for ``(batch, 1, C, T)`` input."""
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (N, 1, C, T), got shape {tuple(x.shape)}")
        if x.shape[1:] != (1, self.channels, self.samples):
            raise ValueError(
                f"Expected input shape (N, 1, {self.channels}, {self.samples}), got {tuple(x.shape)}"
            )
        return self.classifier(self.features(x).flatten(start_dim=1))
