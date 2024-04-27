from sklearn.cluster import MiniBatchKMeans
import torch
import torch.nn as nn
import torch.nn.functional as F

class InferencePalettizedConv2d(nn.Module):
    def __init__(self, lookup_table, weight, bias, stride, dilation, groups, palette_size=256):
        super(InferencePalettizedConv2d, self).__init__()
        if lookup_table is None:
            reshaped = weight.reshape(-1,1).detach().numpy()
            # print(reshaped, "reshaped")
            kmeans = MiniBatchKMeans(n_clusters=min(palette_size,256))
            kmeans.fit(reshaped)
            lookup_table = kmeans.cluster_centers_.squeeze()
            indices = kmeans.labels_
            lookup_table = torch.tensor(lookup_table)
            weight = torch.tensor(indices, dtype=torch.uint8).reshape(weight.shape)
        self.lookup_table = nn.Parameter(lookup_table, requires_grad=False) # dtype arbitrary
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) # same dtype as lookup table
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        # TODO: Add padding

    def forward(self, input):
        full_weights = self.lookup_table[self.weight.to(torch.int32)]
        return F.conv2d(input, full_weights, self.bias, self.stride, 0, self.dilation, self.groups)
