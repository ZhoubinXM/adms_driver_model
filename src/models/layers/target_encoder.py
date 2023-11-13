import torch
import torch.nn as nn
from src.models.layers.base import MLP
# from layers.dla_layer import DLAMLP as MLP
from src.models.layers.dla_layer import DLAMax, DLARepeat
from utils import get_value


class TargetEncoder(nn.Module):

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super(TargetEncoder, self).__init__()
        type = get_value('norm_type')
        # assert type in ['bn', 'ln']
        self.enc = MLP(in_dim,
                       out_dim,
                       norm_dim=49 if type == 'bn' else None,
                       end_with_linear=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)


class TargetVecEncoder(nn.Module):

    def __init__(self, in_channels, hidden_unit, steps, num_subgraph_layers):
        super(TargetVecEncoder, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.layer_seq = nn.Sequential()
        type = get_value('norm_type')
        # assert type in ['bn', 'ln']
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(
                f'lmlp_{i}',
                MLP(in_channels,
                    hidden_unit,
                    norm_dim=49 if type == 'bn' else None))
            in_channels = hidden_unit * 2
        self.max_layer = DLAMax(8, 7, 4)  # magic num: 8, 7, 4, according to 49
        self.repeat_layer = DLARepeat(steps)

    def forward(self, target):
        '''
            Extract target feature from vectorized lane representation

        Args:
            target: [batch size, 1, AGENT_HISTORY_LEN - 1, AGENT_FEATURE_DIM] (vectorized representation)

        Returns:
            x_max: [batch size, max_lane_num, 1, dim]
        '''
        x = target
        for _, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                x = layer(x)
                x = x.transpose(1, -1)
                x_max = self.repeat_layer(self.max_layer(x))
                x = torch.cat([x, x_max], dim=1)
                x = x.transpose(1, -1)
        x_max = self.max_layer(x)
        return x_max
