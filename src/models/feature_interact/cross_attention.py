from typing import Dict, Union
import torch
from src.models.feature_interact.feature_interact import *
from src.models.layers import *


class FICrossAtten(FeatureInteract):
    """Concat Features"""

    def __init__(
        self,
        num_head: int = 4,
        hidden_size: int = 256,
        pooling: str = "max",
        attention: str = "self",
        plot: bool = False,
    ):
        super().__init__()
        self.pooling_method = pooling
        self.attention_method = attention
        self.plot = plot
        self.cross_atten_layer = CrossAttentionLayer(head_num=num_head, d_model=hidden_size, d_ff=4 * hidden_size,
                                                     norm_dim=None, type=self.attention_method)
        self.self_atten_layer = SelfAttentionLayer(head_num=num_head, d_model=hidden_size, d_ff=4 * hidden_size)
        self.apply(weight_init)

    def forward(self, encodings: list[torch.Tensor]):
        """Encodings is tuple with env embedding, env mask, drv_embedding"""

        env_emb, env_mask, drv_emb = encodings

        if len(drv_emb.shape) == 2:
            drv_emb.unsqueeze_(1).unsqueeze_(2)

        if env_emb is None:
            return drv_emb

        if drv_emb is None:
            return env_emb, env_mask
        
        if self.attention_method == "cross":
            fi_features, attention_score = self.cross_atten_layer(env_emb, drv_emb, env_mask.transpose(-2, -1))
            mask = env_mask
        else:
            drv_mask = torch.ones([*env_mask.shape[:3]] + [1]).to(drv_emb.dtype).to(drv_emb.device)
            mask = torch.cat([drv_mask, env_mask], dim=-1)
            env_drv_emb = torch.cat([drv_emb, env_emb], dim=1)
            fi_features, attention_score = self.self_atten_layer(env_drv_emb, mask=mask)
        expaned_mask = mask.reshape(mask.shape[0],
                                    -1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, fi_features.shape[-1])
        fi_features = fi_features * expaned_mask
        # max pooling
        if self.pooling_method == 'max':
            fi_features, _ = torch.max(fi_features, dim=1, keepdim=True)
        # mean pooling
        elif self.pooling_method == 'mean':
            fi_features = torch.sum(fi_features, dim=1, keepdim=True)
            num_agents = torch.sum(mask, dim=-1, keepdim=True)
            fi_features = fi_features / num_agents
            # fi_features = torch.mean(fi_features, dim=1)
        elif self.pooling_method == "first":
            fi_features = fi_features[:, 0].unsqueeze(dim=1)

        if self.plot:
            return fi_features, attention_score
        return fi_features
