from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class KMeansPalettizedLinear(nn.Module):
    def __init__(self, lookup_table, weight, bias, palette_size=256):
        super(KMeansPalettizedLinear, self).__init__()
        if lookup_table is None:
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
