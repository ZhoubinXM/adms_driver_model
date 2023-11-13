import torch
import torch.nn as nn
import math
from copy import deepcopy
from typing import Optional
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.resnet import ResNet50_Weights
from utils import get_value


class Norm(nn.Module):

    def __init__(self, dim: int) -> None:
        super(Norm, self).__init__()
        type = get_value('norm_type')
        # assert type in ['bn', 'ln']
        if type == 'bn':
            self.norm = nn.BatchNorm2d(dim)
        elif type == 'ln':
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.norm, nn.BatchNorm2d):
            # N, P, T, F = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
            # print( P, T, P * T)
            # x = torch.reshape(x, (-1, F, 1, 1))
            # x = self.norm(x)
            # x = torch.reshape(x, (N, P, T, F))
            # method: 0
            # x = x.permute(0, 3, 1, 2) # N, F, P, T
            # x = self.norm(x)
            # x = x.permute(0, 2, 3, 1) # N, P, T, F
            # method: 1
            N, P, T, F = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
            x = torch.reshape(x, (N, P * T, F, 1))
            x = self.norm(x)
            x = torch.reshape(x, (N, P, T, F))
        else:
            x = self.norm(x)
        return x


class MLP(nn.Module):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: Optional[int] = None,
                 norm_dim: Optional[int] = None,
                 end_with_linear: Optional[bool] = False) -> None:
        super(MLP, self).__init__()
        if hidden_features is None:
            hidden_features = out_features
        self.linear1 = nn.Linear(in_features, hidden_features)
        if norm_dim is None:
            norm_dim = hidden_features
        self.norm = Norm(norm_dim)
        self.relu = nn.ReLU()
        if end_with_linear:
            self.linear2 = nn.Linear(hidden_features, out_features)
        self.end_with_linear = end_with_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.norm(x)
        x = self.relu(x)
        if self.end_with_linear:
            x = self.linear2(x)
        return x


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers.
    """
    assert isinstance(module, nn.Module)
    return nn.ModuleList([deepcopy(module) for _ in range(n)])
