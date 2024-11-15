import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings



class KMeansPalettizedLinear(nn.Module):
    def __init__(self, lookup_table, weight, bias, palette_size=256):
        super(KMeansPalettizedLinear, self).__init__()
        if lookup_table is None:
            from sklearn.cluster import KMeans
            reshaped = weight.reshape(-1,1).detach().numpy()
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

    def forward(self, input):
        full_weights = self.lookup_table[self.weight.to(torch.int32)]
        return F.linear(input, full_weights, self.bias)

class FP8PalettizedLinear(nn.Module):
    def __init__(self, lookup_table, weight, bias, palette_size=256):
        super(FP8PalettizedLinear, self).__init__()
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

    def forward(self, input):
        full_weights = self.lookup_table[self.weight.to(torch.int32)]
        return F.linear(input, full_weights, self.bias)

class AffinePalettizedLinear(nn.Module):
    def __init__(self, lookup_table, weight, bias, palette_size=256):
        super(AffinePalettizedLinear, self).__init__()
        if lookup_table is None:
            from sklearn.cluster import KMeans
            reshaped = weight.reshape(-1,1).detach().numpy()
            # print(reshaped, "reshaped")
            min = reshaped.min()
            max = reshaped.max()
            palette = np.append([0], np.linspace(min, max, palette_size-1, dtype=np.float64))
            # print(palette, "palette")
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


    def forward(self, input):
        full_weights = self.lookup_table[self.weight.to(torch.int32)]
        retval = F.linear(input, full_weights, self.bias)        
        return retval

class MinifloatLinear(nn.Module):
    def __init__(self, weight, bias, linear_e4m2=True):
        super(MinifloatLinear, self).__init__()
        if linear_e4m2:
            weight = weight.to(torch.float8_e5m2)
        self.weight = nn.Parameter(weight.to(torch.float8_e4m3fn), requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(torch.float8_e4m3fn), requires_grad=False)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        full_weights = self.weight.to(torch.float32)
        full_bias = self.bias.to(torch.float32) if self.bias is not None else None
        return F.linear(input, full_weights, full_bias)

class SymmetricLinear(nn.Module):
    def __init__(self, weight, bias, palette_size=255, allow_weights_to_flip_signs_in_quantization=False):
        # clamp the weights to the 99th percentile of the absolute value, and then quantize them into a hand-rolled qint8
        # hand-writing this quantization allows us to avoid float8 that isn't in onnxruntime-web as of dec 2024, and allows us to ensure that there are no unfortunate tensor indexing export problems as with affine palettized linear
        if not allow_weights_to_flip_signs_in_quantization:
            assert palette_size < 256, f"weights are stored in an int8 as values between -128 to 127, and your maximum weights will quantize as {palette_size // 2}: your larger weights will flip signs!"
        signed_palette_size = palette_size // 2
        super(SymmetricLinear, self).__init__()
        # find the 99th percentile of the absolute value of the weights
        max_abs = torch.tensor(np.quantile(weight.abs(), 0.99))
        scaling_factor = max_abs / signed_palette_size
        scaled_weights = torch.clamp(weight, -max_abs, max_abs) / scaling_factor
        self.weight = nn.Parameter(scaled_weights.to(torch.int8), requires_grad=False)
        self.scaling_factor = nn.Parameter(scaling_factor, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        full_weights = self.weight.to(torch.float32) * self.scaling_factor
        full_bias = self.bias if self.bias is not None else None
        return F.linear(input, full_weights, full_bias)
