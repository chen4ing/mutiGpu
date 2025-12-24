from typing import Any, cast, Dict, List, Optional, Union
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

__all__ = [
    "checkpoined_vgg",
]

class checkpoined_vgg(nn.Module):
    def flatten_nn_module(self, module: nn.Module) -> List[nn.Module]:
        out = []

        for layer in module:
            if isinstance(layer, nn.ModuleList)\
            or isinstance(layer, nn.Sequential):
                out.extend(self.flatten_nn_module(layer))
            else:
                out.append(layer)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for segment in self.segments.values():
            x = checkpoint(segment, x, use_reentrant=False)
        return x

