import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Any, cast, Dict, List, Optional, Union
from models import vgg19

def flatten_nn_module(module):
    out = []

    for layer in module:
        if isinstance(layer, nn.ModuleList)\
        or isinstance(layer, nn.Sequential):
            out.extend(flatten_nn_module(layer))
        else:
            out.append(layer)
    return out




model = vgg19()
print(len(flatten_nn_module([m for n, m in model._modules.items()])))