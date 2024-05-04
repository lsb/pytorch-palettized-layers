from .palettized_linear import AffinePalettizedLinear
from .palettized_conv2d import AffinePalettizedConv2d
import torch.nn as nn

def palettize_linear_as(m, palette_size=64):
    return AffinePalettizedLinear(None, m.weight.data, (m.bias.data if m.bias is not None else None), palette_size=palette_size)

def palettize_conv2d_as(m, palette_size=256):
    return AffinePalettizedConv2d(None, weight=m.weight.data, bias=(m.bias.data if m.bias is not None else None), stride=m.stride, dilation=m.dilation, groups=m.groups, padding=m.padding, palette_size=palette_size)

def palettize_model(model, linear_palette_size=64, conv_palette_size=256):
    named_children = list(model.named_children())
    for n,m in named_children:
        if isinstance(m, nn.Linear):
            # print("weight count", m.weight.data.reshape(-1).shape[0], n)
            setattr(model, n, palettize_linear_as(m, palette_size=linear_palette_size))
        elif isinstance(m, nn.Conv2d):
            # print("weight count", m.weight.data.reshape(-1).shape[0], n)
            setattr(model, n, palettize_conv2d_as(m, palette_size=conv_palette_size))
        else:
            palettize_model(model=m, linear_palette_size=linear_palette_size, conv_palette_size=conv_palette_size)
