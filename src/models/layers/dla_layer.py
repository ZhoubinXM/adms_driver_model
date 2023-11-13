from typing import Optional
import torch
from torch import nn

import numpy as np

dla_mode = True


class DLAMax(nn.Module):

    def __init__(self, dim1: int, dim2: int, padding: int) -> None:
        super(DLAMax, self).__init__()
        if dla_mode:
            self.max_pool1 = torch.nn.MaxPool2d((dim1, 1),
                                                stride=(dim1, 1),
                                                padding=(padding, 0))
            self.max_pool2 = torch.nn.MaxPool2d((dim2, 1), stride=(dim2, 1))

    def forward(self, x):
        if dla_mode:
            x = self.max_pool1(x)
            x = self.max_pool2(x)
        else:
            x = torch.max(x, -2, keepdim=True)[0]
        return x


class DLARepeat(nn.Module):

    def __init__(self, dim: int) -> None:
        super(DLARepeat, self).__init__()
        self.dim = dim
        if dla_mode:
            self.deconv = torch.nn.ConvTranspose2d(1,
                                                   1,
                                                   kernel_size=(dim, 1),
                                                   stride=(1, 1))

            nn.init.ones_(self.deconv.weight)
            nn.init.zeros_(self.deconv.bias)
            for p in self.deconv.parameters():
                p.requires_grad = False

    def forward(self, x):
        if dla_mode:
            N, C, H, W = x.shape
            x = self.deconv(x.contiguous().view(N * C, 1, 1,
                                                W)).contiguous().view(
                                                    N, C, -1, W)
        else:
            x = x.repeat(1, 1, self.dim, 1)
        return x


class DLALinear(nn.Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        super(DLALinear, self).__init__()
        if dla_mode:
            self.conv = nn.Conv2d(in_features,
                                  out_features,
                                  kernel_size=(1, 1))
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if dla_mode:
            N, C, H, W = x.shape
            x = self.conv(x.transpose(1, -1)).transpose(1, -1)
            # x = self.conv(x.reshape(-1, self.in_features, 1,
            #                         1)).reshape(N, C, H, self.out_features)
        else:
            x = self.linear(x)
        return x


class DLAMLP(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: Optional[int] = None,
                 end_with_linear: Optional[bool] = False) -> None:
        super(DLAMLP, self).__init__()
        if hidden_features is None:
            hidden_features = out_features
        self.conv = nn.Conv2d(in_features, hidden_features, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(hidden_features)
        self.relu = nn.ReLU()
        if end_with_linear:
            self.linear = nn.Conv2d(hidden_features,
                                    out_features,
                                    kernel_size=(1, 1))
        self.in_features = in_features
        self.out_features = out_features
        self.end_with_linear = end_with_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        x = x.reshape(-1, self.in_features, 1, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.end_with_linear:
            x = self.linear(x)
        return x.reshape(N, C, H, self.out_features)


if __name__ == '__main__':
    # test torch.max and DLA_MAX
    input = torch.randn(256, 100, 49, 128)
    torch_max_output = torch.max(input, -2, keepdim=True)[0].numpy()
    dla_max_out = DLAMax(8, 7, 4)(input).numpy()

    np.testing.assert_allclose(torch_max_output, dla_max_out)

    # test DLARepeat
    input = torch.ones(256, 100, 1, 128)
    torch_repeat_output = input.repeat(1, 1, 19, 1).numpy()
    dla_repeat_out = DLARepeat(19)(input).numpy()
    np.testing.assert_allclose(dla_repeat_out, torch_repeat_output)

    # test DLALinear
    input = torch.randn(256, 100, 19, 6)
    torch_linear = nn.Linear(6, 128)
    dla_linear = DLALinear(6, 128)
    torch_linear.weight.data = dla_linear.conv.weight.data.squeeze()
    torch_linear.bias.data = dla_linear.conv.bias.data
    torch_linear_output = torch_linear(input).detach().numpy()
    dla_linear_out = dla_linear(input).detach().numpy()
    np.testing.assert_allclose(dla_linear_out,
                               torch_linear_output,
                               rtol=1e-5,
                               atol=1e-5)
