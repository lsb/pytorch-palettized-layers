from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F

class InferencePalettizedLinear(nn.Module):
    def __init__(self, lookup_table, weight, bias, palette_size=256):
        super(InferencePalettizedLinear, self).__init__()
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
        self.bias = nn.Parameter(bias, requires_grad=False) # same dtype as lookup table

    def forward(self, input):
        full_weights = self.lookup_table[self.weight.to(torch.int32)]
        return F.linear(input, full_weights, self.bias)
