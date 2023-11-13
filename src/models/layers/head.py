from typing import Optional, Union
import torch
import torch.nn as nn
from src.models.layers.base import MLP
from src.models.layers.dla_layer import DLALinear
# from layers.dla_layer import DLAMLP as MLP


class MLPHead(nn.Module):

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hiddend_dim: Optional[int] = None,
                 norm_dim: Optional[int] = None) -> None:
        super(MLPHead, self).__init__()
        # self.linear = DLALinear(in_dim, hiddend_dim)
        self.head = MLP(in_dim,
                        out_dim,
                        hiddend_dim,
                        norm_dim=norm_dim,
                        end_with_linear=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.linear(x)
        return self.head(x)
