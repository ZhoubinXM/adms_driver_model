from typing import Dict, Union
import torch
from src.models.feature_interact.feature_interact import *


class FIConcat(FeatureInteract):
    """Concat Features"""

    def __init__(self):
        super().__init__()
        self.apply(weight_init)

    def forward(self, encodings: list[torch.Tensor]):
        """Encodings is tuple with env embedding, env mask, drv_embedding"""

        env_emb, env_mask, drv_emb = encodings

        if len(drv_emb.shape) == 2:
            drv_emb.unsqueeze_(1).unsqueeze_(2)

        if env_emb is None:
            return drv_emb, env_mask

        if drv_emb is None:
            return env_emb, env_mask
        env_emb = env_emb.max(dim=1)[0].unsqueeze(1)
        return torch.cat([drv_emb, env_emb], dim=1), env_mask
    