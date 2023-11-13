import torch
import torch.nn as nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.resnet import ResNet50_Weights
from src.models.layers.base import clones

class ImageEncoder(nn.Module):

    def __init__(self, d_model: int) -> None:
        super(ImageEncoder, self).__init__()
        self.backbone = resnet_fpn_backbone('resnet50',
                                            trainable_layers=3,
                                            norm_layer=nn.BatchNorm2d,
                                            weights=ResNet50_Weights.DEFAULT)
        self.conv_layers = clones(
            nn.Conv2d(256, 16, kernel_size=(1, 1), stride=(1, 1)), 4)
        self.norms = clones(nn.BatchNorm2d(16), 4)
        self.acts = clones(nn.ReLU(), 4)
        self.linears = nn.ModuleList([
            nn.Linear(61 * 61, d_model),
            nn.Linear(31 * 31, d_model),
            nn.Linear(16 * 16, d_model),
            nn.Linear(8 * 8, d_model),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        res = []
        for v in x.values():
            res.append(v)
        for i in range(4):
            res[i] = self.acts[i](self.norms[i](self.conv_layers[i](res[i])))
            res[i] = torch.flatten(res[i], start_dim=2).unsqueeze(2)
            res[i] = self.linears[i](res[i])
        out = torch.concat(res[:4], dim=1)
        return out
