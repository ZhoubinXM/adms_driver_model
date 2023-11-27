import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, alpha=1., gamma=2.):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        # pt = torch.exp(-CE_loss)
        # self.alpha = self.alpha.to(targets.device).gather(0, targets)
        # F_loss = self.alpha * (1-pt)**self.gamma * CE_loss
        log_p = torch.log(inputs)  # [B, C]
        log_p = log_p.gather(1, targets.reshape(-1, 1)).squeeze()
        p = torch.exp(log_p)
        self.alpha = self.alpha.to(targets.device).gather(0, targets)
        F_loss = -1 * self.alpha * torch.pow(1 - p, self.gamma) * log_p

        return F_loss.sum() / self.alpha.sum()
    