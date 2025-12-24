from typing import Any, cast, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

__all__ = [
    'Linear',
]

hidden_dim = 4096
n_segments = 12

class Linear(nn.Module):
    def __init__(
        self, num_classes: int = 1000, init_weights: bool = True
    ) -> None:
        super().__init__()
        
        self.segments = nn.ModuleDict()

        self.segments[str(0)] = nn.Sequential(
            nn.Flatten(),
            nn.Linear(224 * 224 * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        for segment_id in range(1, n_segments):
            modules = []
            for _ in range(2):
                modules.append(nn.Linear(hidden_dim, hidden_dim))
                modules.append(nn.ReLU(inplace=True))
            if segment_id == n_segments - 1:
                modules.append(nn.Linear(hidden_dim, num_classes))
            self.segments[str(segment_id)] = nn.Sequential(*modules)

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for segment in self.segments.values():
            x = checkpoint(segment, x, use_reentrant=False)
        
        return x


