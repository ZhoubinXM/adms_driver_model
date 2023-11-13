import torch
import torch.nn as nn
from src.models.layers.base import MLP
# from layers.dla_layer import DLAMLP as MLP
from src.models.layers.dla_layer import DLAMax, DLARepeat
from utils import get_value
import torch.nn.functional as F


class AgentEncoder(nn.Module):

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super(AgentEncoder, self).__init__()
        self.enc = MLP(in_dim, out_dim, end_with_linear=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)


class AgentVecEncoder(nn.Module):

    def __init__(self, in_channels, hidden_unit, steps, num_subgraph_layers):
        super(AgentVecEncoder, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.layer_seq = nn.Sequential()
        type = get_value('norm_type')
        assert type in ['bn', 'ln']
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(
                f'lmlp_{i}',
                MLP(in_channels,
                    hidden_unit,
                    norm_dim=1470 if type == 'bn' else None)) # 4900 is magic num
            in_channels = hidden_unit * 2
        self.max_layer = DLAMax(8, 7, 4)  # magic num: 8, 7, 4, according to 49
        self.repeat_layer = DLARepeat(steps)

    def forward(self, agent, mask):
        '''
            Extract lane_feature from vectorized lane representation

        Args:
            lane: [batch size, max_agent_num, step, feature_dim] (vectorized representation)
            mask: [batch size, max_agent_num, step, 1]
        Returns:
            x_max: [batch size, max_agent_num, 1, dim]
        '''
        x = agent
        mask = (mask - 1) * 10000000.0
        for _, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                x = layer(x)
                shortcut = x
                x = x + mask
                x = F.relu(x)
                x = x.transpose(1, -1)
                x_max = self.repeat_layer(self.max_layer(x))
                x = torch.cat([shortcut.transpose(1, -1), x_max], dim=1)
                x = x.transpose(1, -1)
        x_max = self.max_layer(x)
        return x_max
