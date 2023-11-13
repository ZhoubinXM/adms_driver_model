import abc
import torch
from torch import nn
from typing import Dict, Tuple, Optional, List, Any
from src.utils import weight_init


class Encoder(nn.Module):
    """Base class for feature encoder"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, inputs_0, inputs_1, sparse_feature, dense_feature):
        N0, C0, H0, W0 = inputs_0.shape  # [B, 31, 49, 10]
        N1, C1, H1, W1 = inputs_1.shape  # [B, 75, 19, 7]
        target = inputs_0[0:N0, 0:1, 0:H0, 0:9]
        agent = inputs_0[0:N0, 1:C0, 0:H0, 0:9]
        agent_mask = inputs_0[0:N0, 0:C0, 0:H0, 9:W0]
        lane = inputs_1[0:N1, 0:C1, 0:H1, 0:6]
        lane_mask = inputs_1[0:N1, 0:C1, 0:H1, 6:W1]
        return self._forward(target=target, agent=agent, agent_mask=agent_mask, lane=lane, lane_mask=lane_mask,
                             sparse_feature=sparse_feature, dense_feature=dense_feature, target_pos_in_ego=None,
                             img=None)

    @abc.abstractmethod
    def _forward(self, *args, **kwargs) -> Any:
        """
        Abstract method for forward pass. Returns dictionary of encodings.

        :param inputs: Dictionary with ...
        :return encodings: Dictionary with input encodings
        """
        raise NotImplementedError()