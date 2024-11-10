from pytorch_palettized_layers.palettized_linear import KMeansPalettizedLinear, MinifloatLinear
from pytorch_palettized_layers.palettized_conv2d import KMeansPalettizedConv2d, MinifloatConv2d
from pytorch_palettized_layers import palettize_model, minifloat_model
import torch
import torch.nn as nn

class TwoLayerModel(nn.Module):
    def __init__(self):
        super(TwoLayerModel, self).__init__()
        self.linear1 = nn.Linear(3, 4)
        self.linear2 = nn.Linear(4, 2)
        self.relu = nn.LeakyReLU(negative_slope=0.5)
    
    def forward(self, x):
        print(x)
        x = self.linear1(x)
        print(x)
        x = self.relu(x)
        print(x)
        x = self.linear2(x)
        print(x)
        return x

class Conv2dModel(nn.Module):
    def __init__(self):
        super(Conv2dModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 2)
        self.conv2 = nn.Conv2d(4, 2, 2)
        self.relu = nn.LeakyReLU(negative_slope=0.5)
    
    def forward(self, x):
        print(x)
        x = self.conv1(x)
        print(x)
        x = self.relu(x)
        print(x)
        x = self.conv2(x)
        print(x)
        return x

def make_vanilla_linear():
    vanilla_model = TwoLayerModel()
    vanilla_model.linear1.weight.data = (6 - torch.arange(12)).reshape_as(vanilla_model.linear1.weight.data).to(torch.float32)
    vanilla_model.linear1.bias.data = torch.arange(4).to(torch.float32)
    vanilla_model.linear2.weight.data = (4 - torch.arange(8)).reshape_as(vanilla_model.linear2.weight.data).to(torch.float32)
    vanilla_model.linear2.bias.data = torch.arange(2).to(torch.float32)
    return vanilla_model

def make_vanilla_conv2d():
    vanilla_model = Conv2dModel()
    print(vanilla_model.conv1.weight.data)
    vanilla_model.conv1.weight.data = (6 - torch.arange(24)).reshape_as(vanilla_model.conv1.weight.data).to(torch.float32)
    vanilla_model.conv1.bias.data = torch.arange(4).to(torch.float32)
    vanilla_model.conv2.weight.data = (4 - torch.arange(8)).reshape_as(vanilla_model.conv2.weight.data).to(torch.float32)
    vanilla_model.conv2.bias.data = torch.arange(2).to(torch.float32)
    return vanilla_model

def test_linear():
    sample_input = torch.tensor([1, 2, 3], dtype=torch.float32)
    vanilla_model = make_vanilla_linear()
    vanilla_output = vanilla_model(sample_input)

    # Create a palettized model from the vanilla model

    palettized_model = make_vanilla_linear()
    palettized_model.linear1 = KMeansPalettizedLinear(
        None,
        vanilla_model.linear1.weight.data,
        vanilla_model.linear1.bias.data,
        palette_size=12
    )
    palettized_model.linear2 = KMeansPalettizedLinear(
        None,
        vanilla_model.linear2.weight.data,
        vanilla_model.linear2.bias.data,
        palette_size=8
    )

    # Compare the outputs of the vanilla and palettized models

    palettized_output = palettized_model(sample_input)
    print(vanilla_output)
    print(palettized_output)
    assert torch.allclose(vanilla_output, palettized_output)

    # Create a minifloat model from the vanilla model

    fp8_model = make_vanilla_linear()
    fp8_model.linear1 = MinifloatLinear(
        vanilla_model.linear1.weight.data,
        vanilla_model.linear1.bias.data
    )
    fp8_model.linear2 = MinifloatLinear(
        vanilla_model.linear2.weight.data,
        vanilla_model.linear2.bias.data
    )

    # Compare the outputs of the vanilla and minifloat models

    fp8_output = fp8_model(sample_input)
    print(vanilla_output)
    print(fp8_output)
    assert torch.allclose(vanilla_output, fp8_output)

    # Test the palettize_model function

    palettized_model = make_vanilla_linear()
    palettize_model(palettized_model, linear_palette_size=77 + 1 + 1, conv_palette_size=256)
    palettized_output = palettized_model(sample_input)
    print(vanilla_output)
    print(palettized_output)
    assert torch.allclose(vanilla_output, palettized_output)

    # Test the minifloat_model function

    fp8_model = make_vanilla_linear()
    minifloat_model(fp8_model)
    fp8_output = fp8_model(sample_input)
    print(vanilla_output)
    print(fp8_output)
    assert torch.allclose(vanilla_output, fp8_output)

def test_conv2d():
    vanilla_model = Conv2dModel()
    sample_input = torch.ones(1, 3, 5, 5)

    # Create a palettized model from the vanilla model
    palettized_model = Conv2dModel()
    palettized_model.conv1 = KMeansPalettizedConv2d(
        None,
        weight=vanilla_model.conv1.weight.data,
        bias=vanilla_model.conv1.bias.data,
        stride=vanilla_model.conv1.stride,
        dilation=vanilla_model.conv1.dilation,
        groups=vanilla_model.conv1.groups,
        padding=vanilla_model.conv1.padding,
        palette_size=48
    )
    palettized_model.conv2 = KMeansPalettizedConv2d(
        None,
        weight=vanilla_model.conv2.weight.data,
        bias=vanilla_model.conv2.bias.data,
        stride=vanilla_model.conv2.stride,
        dilation=vanilla_model.conv2.dilation,
        groups=vanilla_model.conv2.groups,
        padding=vanilla_model.conv2.padding,
        palette_size=32
    )

    # Create a minifloat model from the vanilla model
    fp8_model = Conv2dModel()
    fp8_model.conv1 = MinifloatConv2d(
        vanilla_model.conv1.weight.data,
        vanilla_model.conv1.bias.data,
        vanilla_model.conv1.stride,
        vanilla_model.conv1.dilation,
        vanilla_model.conv1.groups,
        vanilla_model.conv1.padding
    )
    fp8_model.conv2 = MinifloatConv2d(
        vanilla_model.conv2.weight.data,
        vanilla_model.conv2.bias.data,
        vanilla_model.conv2.stride,
        vanilla_model.conv2.dilation,
        vanilla_model.conv2.groups,
        vanilla_model.conv2.padding
    )

    # Compare the outputs of the vanilla and palettized models
    vanilla_output = vanilla_model(sample_input)
    palettized_output = palettized_model(sample_input)
    print("vanilla output", vanilla_output)
    print("palettized_output", palettized_output)
    assert torch.allclose(vanilla_output, palettized_output)

    # Compare the outputs of the vanilla and minifloat models
    fp8_output = fp8_model(sample_input)
    print("vanilla output", vanilla_output)
    print("fp8_output", fp8_output)
    assert torch.allclose(vanilla_output, fp8_output, atol=1e-1, rtol=1e-1)


