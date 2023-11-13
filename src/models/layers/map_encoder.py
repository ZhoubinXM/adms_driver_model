import torch
from torch import nn
from src.models.layers.base import MLP
# from layers.dla_layer import DLAMLP as MLP
from src.models.layers.dla_layer import DLAMax, DLARepeat
from utils import get_value
import torch.nn.functional as F


class MapEncoder(nn.Module):

    def __init__(self, in_channels, hidden_unit, steps, num_subgraph_layers):
        super(MapEncoder, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.layer_seq = nn.Sequential()
        type = get_value('norm_type')
        assert type in ['bn', 'ln']
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(
                f'lmlp_{i}',
                MLP(in_channels,
                    hidden_unit,
                    norm_dim=1425 if type == 'bn' else None)) # 3800 is magic num
            in_channels = hidden_unit * 2
        self.max_layer = DLAMax(8, 3, 3)  # magic num: 8, 3, 3, according to 19
        self.repeat_layer = DLARepeat(steps)

    def forward(self, lane, mask):
        '''
            Extract lane_feature from vectorized lane representation

        Args:
            lane: [batch size, max_lane_num, 9, 7] (vectorized representation)

        Returns:
            x_max: [batch size, max_lane_num, 1, dim]
        '''
        x = lane
        mask = (mask - 1) * 10000000.0
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                # x [bs,max_lane_num,19,dim]
                x = layer(x)
                shortcut = x
                x = x + mask
                x = F.relu(x)
                x = x.transpose(1, -1)
                x_max = self.repeat_layer(self.max_layer(x))
                x = torch.cat([shortcut.transpose(1, -1), x_max], dim=1)
                x = x.transpose(1, -1)
        # x_max = torch.max(x, -2, keepdim=True)[0]
        x_max = self.max_layer(x)
        return x_max
