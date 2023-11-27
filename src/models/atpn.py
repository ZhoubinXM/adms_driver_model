from lightning.pytorch.utilities.types import STEP_OUTPUT
import os
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import src.models.encoder.encoder as enc
import src.models.feature_interact.feature_interact as fi
import src.models.decoder.decoder as dec
from src.models.layers.focal_loss import FocalLoss
from typing import Any, Dict, Tuple, Optional
import pandas as pd
import numpy as np

# metrics
from torchmetrics import Metric, MinMetric, MeanMetric
from torchmetrics.regression import MeanAbsoluteError, R2Score, MeanAbsolutePercentageError
from torchmetrics.classification import Accuracy, Recall, Precision, ConfusionMatrix, F1Score
from sklearn.metrics import confusion_matrix, classification_report

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
        torch.autograd.set_detect_anomaly(True)
        self.weight_decay = kwargs['weight_decay']
        self.lr_decay = kwargs['lr_decay']
        self.lr = kwargs['lr']
        self.lbl_proc = kwargs['lbl_proc']
        self.step_size = kwargs['step_size']
        self.mode = kwargs['mode']

        assert self.lbl_proc in ['log', 'cbrt', 'none']
        assert self.mode in ['reg', 'cls', 'equal_width_cls', 'quantile_cls', 'none']

        self.encoder = encoder
        self.fi = feature_interact
        self.decoder = decoder
        if self.mode == "reg":
            self.criterion = nn.SmoothL1Loss(beta=1.0)
            self.metrics: list[Metric] = [MeanAbsoluteError(), MeanAbsolutePercentageError(), R2Score()]
            # self.criterion = self.quantile_loss
        elif 'cls' in self.mode:
            # self.criterion = nn.CrossEntropyLoss()
            self.tot_criterion = FocalLoss(alpha=torch.tensor([0.04, 0.15, 0.28, 0.38, 0.15]), gamma=2)
            self.tob_criterion = FocalLoss(alpha=torch.tensor([0.1, 0.1, 0.13, 0.2, 0.32, 0.15]), gamma=2)
            self.metrics: list[Metric] = [
                Accuracy(task="multiclass", num_classes=10),
                Precision(task="multiclass", average='macro', num_classes=10),
                Recall(task="multiclass", average='macro', num_classes=10),
                F1Score(task="multiclass", num_classes=10)
            ]
        self.attention_scores = None
        self.output_dir = kwargs['output_dir']
        self.test_out = None
        if self.lbl_proc == 'quantile':
            from src.data.adms_dataset import QuantileTransformer
            self.tot_qt = QuantileTransformer()
            self.tob_qt = QuantileTransformer()
            all_quantile = pd.read_pickle("./data/all_quantile.pkl")
            self.tot_qt.fit(torch.tensor(all_quantile.tot.values))
            self.tob_qt.fit(torch.tensor(all_quantile.tob.values))

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass for prediction model
        :param inputs: Dictionary with ...
            ...
        :return outputs: Prediction Result
        """
        encodings = self.encoder(*args, **kwargs)
        encodings = self.fi(encodings)
        if isinstance(encodings, tuple):
            self.attention_scores = encodings[1]
            encodings = encodings[0]
        outputs = self.decoder(encodings)

        return outputs

    def model_step(self, data):
        """main model step"""
        inputs = self._organize_inputs(data)
        pred_tot, pred_tob = self(**inputs)
        gt_tot, gt_tob = data[5], data[6]
        gt_tot, gt_tob = self.proc_label(gt_tot, gt_tob)
        if self.mode == 'reg':
            pred_tot = pred_tot.view(-1)
            pred_tob = pred_tob.view(-1)
        elif 'cls' in self.mode:
            gt_tot = gt_tot.long()
            gt_tob = gt_tob.long()
            pred_tot = pred_tot.reshape(pred_tot.shape[0], -1)
            pred_tob = pred_tob.reshape(pred_tob.shape[0], -1)
        # Loss
        if self.mode == 'equal_width_cls':
            tot_loss = self.tot_criterion(pred_tot, gt_tot)
            tob_loss = self.tob_criterion(pred_tob, gt_tob)
        else:
            tot_loss = self.criterion(pred_tot, gt_tot)
            tob_loss = self.criterion(pred_tob, gt_tob)
        loss = tot_loss + tob_loss
        # test loop return
        if self.test_out is not None and self.lbl_proc == 'quantile':
            gt_tot_quantile, gt_tob_quantile, pred_tot_quantile, pred_tob_quantile = \
                gt_tot, gt_tob, pred_tot, pred_tob
        # metrics
        gt_tot, gt_tob, pred_tot, pred_tob = self.inv_label(gt_tot, gt_tob, pred_tot, pred_tob)
        tot_metrics, tob_metrics = [], []
        for metric in self.metrics:
            metric.to(pred_tob.device)
            tot_metric = metric(pred_tot, gt_tot)
            tob_metric = metric(pred_tob, gt_tob)
            tot_metrics.append(tot_metric)
            tob_metrics.append(tob_metric)
        # test loop return
        if self.test_out is not None:
            self.test_out.append(
                torch.stack(
                    [
                        gt_tot,
                        gt_tob,
                        pred_tot,
                        pred_tob,
                        # gt_tot_quantile.cpu(),
                        # gt_tob_quantile.cpu(),
                        # pred_tot_quantile.cpu(),
                        # pred_tob_quantile.cpu()
                    ],
                    dim=-1).cpu().numpy())
        return [loss, tot_loss, tob_loss], [tot_metrics, tob_metrics]

    def training_step(self, data, batch_idx):
        losses, metrics = self.model_step(data)
        self.log('train/loss', losses[0], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/tot_loss', losses[1], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/tob_loss', losses[2], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        if self.mode == 'reg':
            self.log('train/tot_mae', metrics[0][0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('train/tob_mae', metrics[1][0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('train/tot_mape', metrics[0][1], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('train/tob_mape', metrics[1][1], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('train/tot_r2', metrics[0][2], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('train/tob_r2', metrics[1][2], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        elif 'cls' in self.mode:
            self.log('train/tot_acc', metrics[0][0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('train/tob_acc', metrics[1][0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('train/tot_prc', metrics[0][1], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('train/tob_prc', metrics[1][1], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('train/tot_recall', metrics[0][2], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('train/tob_recall', metrics[1][2], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('train/tot_f1', metrics[0][3], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('train/tob_f1', metrics[1][3], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        if self.attention_scores is not None and self.global_step == batch_idx + (self.current_epoch * 108):
            self.plot_attention(self.attention_scores)
        return losses[0]

    def validation_step(self, data, batch_idx):
        losses, metrics = self.model_step(data)
        self.log('val/loss', losses[0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/tot_loss', losses[1], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/tob_loss', losses[2], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        if self.mode == 'reg':
            self.log('val/tot_mae', metrics[0][0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val/tob_mae', metrics[1][0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val/tot_mape', metrics[0][1], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val/tob_mape', metrics[1][1], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val/tot_r2', metrics[0][2], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val/tob_r2', metrics[1][2], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        elif 'cls' in self.mode:
            self.log('val/tot_acc', metrics[0][0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val/tob_acc', metrics[1][0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val/tot_prc', metrics[0][1], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val/tob_prc', metrics[1][1], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val/tot_recall', metrics[0][2], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val/tob_recall', metrics[1][2], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val/tot_f1', metrics[0][3], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val/tob_f1', metrics[1][3], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, data, batch_idx):
        losses, metrics = self.model_step(data)
        self.log('test/loss', losses[0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/tot_loss', losses[1], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/tob_loss', losses[2], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        if self.mode == 'reg':
            self.log('test/tot_mae', metrics[0][0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('test/tob_mae', metrics[1][0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('test/tot_mape', metrics[0][1], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('test/tob_mape', metrics[1][1], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('test/tot_r2', metrics[0][2], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('test/tob_r2', metrics[1][2], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        elif 'cls' in self.mode:
            self.log('test/tot_acc', metrics[0][0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('test/tob_acc', metrics[1][0], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('test/tot_prc', metrics[0][1], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('test/tob_prc', metrics[1][1], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('test/tot_recall', metrics[0][2], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('test/tob_recall', metrics[1][2], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('test/tot_f1', metrics[0][3], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log('test/tob_f1', metrics[1][3], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def on_test_epoch_start(self) -> None:
        from src.eval import log
        self._logger = log
        self.test_out = []

    def on_test_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if self.attention_scores is not None:
            save_dir = os.path.join(self.output_dir, f"batch_{batch_idx+1}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.save_attention_plots(self.attention_scores, save_dir, batch[0])

    def on_test_epoch_end(self) -> None:
        pred_res = np.concatenate(self.test_out, axis=0)
        df = pd.DataFrame(
            pred_res, columns=[
                'gt_tot', 'gt_tob', 'pred_tot', 'pred_tob', 
                # 'gt_tot_quantile', 'gt_tob_quantile',
                # 'pred_tot_quantile', 'pred_tob_quantile'
            ])
        if 'cls' in self.mode:
            gt_tot, gt_tob, pred_tot, pred_tob = df.gt_tot, df.gt_tob, df.pred_tot, df.pred_tob
            self._logger.info(f'[TOT][cls report]: \n{classification_report(gt_tot, pred_tot)}')
            self._logger.info(f'[TOT][confusion matrix]: \n{confusion_matrix(gt_tot, pred_tot)}')
            self._logger.info(f'[TOB][cls report]: \n{classification_report(gt_tob, pred_tob)}')
            self._logger.info(f'[TOB][confusion matrix]: \n{confusion_matrix(gt_tob, pred_tob)}')
        if self.test_out is not None:
            save_dir = os.path.join(self.output_dir, "pred_res.csv")
            df.to_csv(save_dir, index=False)
            self._logger.info(f"save pred res to {save_dir}")

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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=self.step_size, gamma=self.lr_decay)

        return [optimizer], [scheduler]

    def _organize_inputs(self, data):
        inputs = {'inputs_0': data[1], 'inputs_1': data[2], 'sparse_feature': data[3], 'dense_feature': data[4]}
        return inputs

    def proc_label(self, gt_tot, gt_tob):
        if self.lbl_proc == 'log':
            gt_tot = torch.log(gt_tot + 1)
            gt_tob = torch.log(gt_tob + 1)
        elif self.lbl_proc == 'cbrt':
            gt_tot = torch.pow(gt_tot, 1 / 3)
            gt_tob = torch.pow(gt_tob, 1 / 3)
        elif self.mode == 'equal_width_cls':
            gt_tot = torch.where(gt_tot > 4, torch.tensor(4), gt_tot)
            gt_tob = torch.where(gt_tob > 5, torch.tensor(5), gt_tob)

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
        elif self.lbl_proc == "quantile":
            gt_tot = self.tot_qt.inverse_transform(gt_tot.cpu())
            gt_tob = self.tob_qt.inverse_transform(gt_tob.cpu())
            pred_tot = self.tot_qt.inverse_transform(pred_tot.cpu())
            pred_tob = self.tob_qt.inverse_transform(pred_tob.cpu())
        elif 'cls' in self.mode:
            _, pred_tot = torch.max(pred_tot, dim=-1)
            _, pred_tob = torch.max(pred_tob, dim=-1)

        return gt_tot, gt_tob, pred_tot, pred_tob

    def plot_attention(self, attention_scores):
        attention_scores = attention_scores[:2]
        # attention_scores: shape (batch_size, num_heads, sequence_length, sequence_length)
        batch_size, num_heads, seq_len, _ = attention_scores.shape
        for i in range(batch_size):
            for j in range(num_heads):
                grid = attention_scores[i, j].unsqueeze(0)
                self.logger.experiment.add_image(f"sample_{i}_attention_scores/head_{j}", grid,
                                                 global_step=self.global_step)

    def save_attention_plots(self, attention_scores, directory, uuids):
        # attention_scores: shape (batch_size, num_heads, sequence_length, sequence_length)
        batch_size, num_heads, seq_len, _ = attention_scores.shape
        for i in range(batch_size):
            fig, axs = plt.subplots(1, num_heads, figsize=(15, 4))
            fig.suptitle(f'uuid: {uuids[i]}')
            for j in range(num_heads):
                # Draw only the first row of the attention scores
                im = axs[j].imshow(attention_scores[i, j, 0, :].reshape(1, -1).cpu().detach().numpy(), cmap='viridis',
                                   aspect='auto')
                axs[j].set_title(f'Head {j+1}')
                axs[j].set_yticks([])  # Hide the y-axis
                # Draw colorbar
                divider = make_axes_locatable(axs[j])
                cax = divider.append_axes("bottom", size="5%", pad=0.35)
                plt.colorbar(im, cax=cax, orientation='horizontal')
            plt.savefig(f'{directory}/{uuids[i]}.jpg')
            plt.close(fig)  # Close the figure to free up memory

    def quantile_loss(self, f, y):
        q = 0.9
        e = (y - f)
        return torch.mean(torch.max(q * e, (q - 1) * e))
