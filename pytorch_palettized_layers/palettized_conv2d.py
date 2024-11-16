import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KMeansPalettizedConv2d(nn.Module):
    def __init__(self, lookup_table, weight, bias, stride, dilation, groups, padding, palette_size=256):
        super(KMeansPalettizedConv2d, self).__init__()
        if lookup_table is None:
            from sklearn.cluster import KMeans
            reshaped = weight.reshape(-1,1).detach().numpy()
            # print(reshaped, "reshaped")
            kmeans = KMeans(n_clusters=min(palette_size,256))
            kmeans.fit(reshaped)
            lookup_table = kmeans.cluster_centers_.squeeze()
            indices = kmeans.labels_
            lookup_table = torch.tensor(lookup_table)
            weight = torch.tensor(indices, dtype=torch.uint8).reshape(weight.shape)
        self.lookup_table = nn.Parameter(lookup_table, requires_grad=False) # dtype arbitrary
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False) # same dtype as lookup table
        else:
            self.register_parameter('bias', None)
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        # TODO: Add padding

    def forward(self, input):
        full_weights = self.lookup_table[self.weight.to(torch.int32)]
        return F.conv2d(input, full_weights, self.bias, self.stride, self.padding, self.dilation, self.groups)

class FP8PalettizedConv2d(nn.Module):
    def __init__(self, lookup_table, weight, bias, stride, dilation, groups, padding):
        super(FP8PalettizedConv2d, self).__init__()
        if lookup_table is None:
            from sklearn.cluster import KMeans
            reshaped = weight.reshape(-1,1).detach().cpu()
            fp8weights = np.unique(reshaped.detach().cpu().to(torch.float8_e4m3fn).to(torch.float64).numpy())
            k = KMeans(n_clusters=len(fp8weights))
            k.fit(np.arange(len(fp8weights)).reshape(-1,1))
            k.cluster_centers_ = fp8weights.reshape(-1,1)
            indices = k.predict(reshaped.numpy().astype(np.float64))
            assert indices.max() < 256
            lookup_table = torch.tensor(fp8weights).to(weight.dtype)
            weight = torch.tensor(indices, dtype=torch.uint8).reshape(weight.shape)
        self.lookup_table = nn.Parameter(lookup_table, requires_grad=False) # dtype arbitrary
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False) # same dtype as lookup table
        else:
            self.register_parameter('bias', None)
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding

    def forward(self, input):
        full_weights = self.lookup_table[self.weight.to(torch.int32)]
        return F.conv2d(input, full_weights, self.bias, self.stride, self.padding, self.dilation, self.groups)

class AffinePalettizedConv2d(nn.Module):
    def __init__(self, lookup_table, weight, bias, stride, dilation, groups, padding, palette_size=256):
        super(AffinePalettizedConv2d, self).__init__()
        if lookup_table is None:
            from sklearn.cluster import KMeans
            reshaped = weight.reshape(-1,1).detach().numpy()
            min = reshaped.min()
            max = reshaped.max()
            palette = np.append([0], np.linspace(min, max, palette_size-1, dtype=np.float64))
            k = KMeans(n_clusters=len(palette))
            k.fit(np.arange(len(palette)).reshape(-1,1))
            k.cluster_centers_ = palette.reshape(-1,1)
            indices = k.predict(reshaped.astype(np.float64))
            assert indices.max() < 256
            lookup_table = torch.tensor(palette).to(weight.dtype)
            weight = torch.tensor(indices, dtype=torch.uint8).reshape(weight.shape)

        self.lookup_table = nn.Parameter(lookup_table, requires_grad=False) # dtype arbitrary
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False) # same dtype as lookup table
        else:
            self.register_parameter('bias', None)
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding

    def forward(self, input):
        full_weights = self.lookup_table[self.weight.to(torch.int32)]
        return F.conv2d(input, full_weights, self.bias, self.stride, self.padding, self.dilation, self.groups)

class MinifloatConv2d(nn.Module):
    def __init__(self, weight, bias, stride, dilation, groups, padding):
        super(MinifloatConv2d, self).__init__()
        self.weight = nn.Parameter(weight.to(torch.float8_e4m3fn), requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(torch.float8_e4m3fn), requires_grad=False)
        else:
            self.register_parameter('bias', None)
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding

    def forward(self, input):
        full_weights = self.weight.to(torch.float32)
        full_bias = self.bias.to(torch.float32) if self.bias is not None else None
        return F.conv2d(input.to(torch.float32), full_weights, full_bias, self.stride, self.padding, self.dilation, self.groups)

class SymmetricConv2d(nn.Module):
    def __init__(self, weight, bias, stride, dilation, groups, padding, palette_size=255, allow_weights_to_flip_signs_in_quantization=False):
        super(SymmetricConv2d, self).__init__()
        if not allow_weights_to_flip_signs_in_quantization:
            assert palette_size < 256, f"weights are stored in an int8 as values between -128 to 127, and your maximum weights will quantize as {palette_size // 2}: your larger weights will flip signs!"
        signed_palette_size = palette_size // 2
        max_abs = weight.abs().max()
        scaling_factor = max_abs / signed_palette_size
        scaled_weights = torch.round(torch.clamp(weight, -max_abs, max_abs) / scaling_factor)
        self.weight = nn.Parameter(scaled_weights.to(torch.int8), requires_grad=False)
        self.scaling_factor = nn.Parameter(scaling_factor, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter('bias', None)
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding

    def forward(self, input):
        full_weights = self.weight.to(torch.float32) * self.scaling_factor
        full_bias = self.bias if self.bias is not None else None
        return F.conv2d(input.to(torch.float32), full_weights, full_bias, self.stride, self.padding, self.dilation, self.groups)
