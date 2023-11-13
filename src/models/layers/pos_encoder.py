import torch
import torch.nn as nn
from src.models.layers.base import MLP
# from layers.dla_layer import DLAMLP as MLP



class PosEncoder(nn.Module):

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super(PosEncoder, self).__init__()
        self.enc = MLP(in_dim, out_dim, end_with_linear=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)
