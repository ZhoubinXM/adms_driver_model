import torch
import torch.nn as nn
import lightning.pytorch as pl

import src.models.encoder.encoder as enc
import src.models.feature_interact.feature_interact as fi
import src.models.decoder.decoder as dec
from typing import Any, Dict, Tuple, Optional

# metrics
from torchmetrics import Metric, MinMetric, MeanMetric
from torchmetrics.regression import MeanAbsoluteError, R2Score, MeanAbsolutePercentageError


class ATPN(pl.LightningModule):
    """
    ADMS prediction model
    """

    def __init__(self, encoder: enc.Encoder, feature_interact: fi.FeatureInteract, decoder: dec.Decoder, *args,
                 **kwargs):
        """
        Initializes model for ADMS prediction task
        """
        super().__init__()
        self.weight_decay = kwargs['weight_decay']
        self.lr_decay = kwargs['lr_decay']
        self.lr = kwargs['lr']
        self.lbl_proc = kwargs['lbl_proc']
        self.encoder = encoder
        self.fi = feature_interact
        self.decoder = decoder
        self.criterion = nn.SmoothL1Loss(beta=1.0)
        self.metrics: list[Metric] = [MeanAbsoluteError(), MeanAbsolutePercentageError(), R2Score()]

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass for prediction model
        :param inputs: Dictionary with ...
            ...
        :return outputs: Prediction Result
        """
        encodings = self.encoder(*args, **kwargs)
        encodings = self.fi(encodings)
        outputs = self.decoder(encodings)

        return outputs

    def model_step(self, data):
        """main model step"""
        inputs = self._organize_inputs(data)
        pred_tot, pred_tob = self(**inputs)
        gt_tot, gt_tob = data[4], data[5]
        gt_tot, gt_tob = self.proc_label(gt_tot, gt_tob)
        pred_tot = pred_tot.view(-1)
        pred_tob = pred_tob.view(-1)
        # Loss
        tot_loss = self.criterion(pred_tot, gt_tot)
        tob_loss = self.criterion(pred_tob, gt_tob)
        loss = tot_loss + tob_loss
        # metrics
        gt_tot, gt_tob, pred_tot, pred_tob = self.inv_label(gt_tot, gt_tob, pred_tot, pred_tob)
        tot_metrics, tob_metrics = [], []
        for metric in self.metrics:
            metric.to(pred_tob.device)
            tot_metric = metric(pred_tot, gt_tot)
            tob_metric = metric(pred_tob, gt_tob)
            tot_metrics.append(tot_metric)
            tob_metrics.append(tob_metric)
        return [loss, tot_loss, tob_loss], [tot_metrics, tob_metrics]

    def training_step(self, data, batch_idx):
        losses, metrics = self.model_step(data)
        self.log('train/loss', losses[0], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/tot_loss', losses[1], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/tob_loss', losses[2], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/tot_mae', metrics[0][0], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/tob_mae', metrics[1][0], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/tot_mape', metrics[0][1], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/tob_mape', metrics[1][1], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/tot_r2', metrics[0][2], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/tob_r2', metrics[1][2], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return losses[0]

    def validation_step(self, data, batch_idx):
        losses, metrics = self.model_step(data)
        self.log('val/loss', losses[0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/tot_loss', losses[1], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/tob_loss', losses[2], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/tot_mae', metrics[0][0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/tob_mae', metrics[1][0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/tot_mape', metrics[0][1], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/tob_mape', metrics[1][1], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/tot_r2', metrics[0][2], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/tob_r2', metrics[1][2], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, data, batch_idx):
        losses, metrics = self.model_step(data)
        self.log('test/loss', losses[0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/tot_loss', losses[1], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/tob_loss', losses[2], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/tot_mae', metrics[0][0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/tob_mae', metrics[1][0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/tot_mape', metrics[0][1], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/tob_mape', metrics[1][1], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/tot_r2', metrics[0][2], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/tob_r2', metrics[1][2], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d, nn.Embedding)
        blacklist_weight_modules = (nn.BatchNorm2d, nn.LayerNorm)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                need_param = True
                if need_param:
                    if 'bias' in param_name:
                        no_decay.add(full_param_name)
                    elif 'weight' in param_name:
                        if isinstance(module, whitelist_weight_modules):
                            decay.add(full_param_name)
                        elif isinstance(module, blacklist_weight_modules):
                            no_decay.add(full_param_name)
                    elif not ('weight' in param_name or 'bias' in param_name):
                        no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [param_dict[param_name] for param_name in sorted(list(decay))],
                "weight_decay": self.weight_decay
            },
            {
                "params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
                "weight_decay": 0.0
            },
        ]

        optimizer = torch.optim.Adam(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=40, gamma=self.lr_decay)

        return [optimizer], [scheduler]

    def _organize_inputs(self, data):
        inputs = {'inputs_0': data[0], 'inputs_1': data[1], 'sparse_feature': data[2], 'dense_feature': data[3]}
        return inputs

    def proc_label(self, gt_tot, gt_tob):
        if self.lbl_proc == 'log':
            gt_tot = torch.log(gt_tot + 1)
            gt_tob = torch.log(gt_tob + 1)
        elif self.lbl_proc == 'cbrt':
            gt_tot = torch.pow(gt_tot, 1 / 3)
            gt_tob = torch.pow(gt_tob, 1 / 3)

        return gt_tot, gt_tob

    def inv_label(self, gt_tot, gt_tob, pred_tot, pred_tob):
        if self.lbl_proc == 'log':
            gt_tot = torch.exp(gt_tot) - 1
            gt_tob = torch.exp(gt_tob) - 1
            pred_tot = torch.exp(pred_tot) - 1
            pred_tob = torch.exp(pred_tob) - 1
        elif self.lbl_proc == 'cbrt':
            gt_tot = torch.pow(gt_tot, 3)
            gt_tob = torch.pow(gt_tob, 3)
            pred_tot = torch.pow(pred_tot, 3)
            pred_tob = torch.pow(pred_tob, 3)

        return gt_tot, gt_tob, pred_tot, pred_tob
