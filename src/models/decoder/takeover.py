from typing import Dict, Union
import torch
from src.models.decoder.decoder import *
from src.models.layers import MLPHead


class TakeOverDecoder(Decoder):
    """TOT, TOB, LT prediction head"""

    def __init__(self, input_dim: int = 256,
                 output_dim: int = 1):
        super().__init__()
        self.dense = nn.Linear(512, input_dim)
        self.tot_head = MLPHead(input_dim,
                                output_dim,
                                2 * input_dim,
                                norm_dim=None)
        self.tob_head = MLPHead(input_dim,
                                output_dim+1,
                                2 * input_dim,
                                norm_dim=None)
        self.apply(weight_init)
        self.output_dim = output_dim
        
    def forward(self, encoding):
        if isinstance(encoding, tuple):
            encoding, mask = encoding[0], encoding[1]
            encoding = encoding.reshape(encoding.shape[0], 1, 1, -1)
            encoding = self.dense(encoding)
        if self.output_dim != 1:
            return torch.softmax(self.tot_head(encoding), dim=-1), \
                    torch.softmax(self.tob_head(encoding), dim=-1)
        return self.tot_head(encoding), self.tob_head(encoding)