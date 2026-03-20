import io 
import matplotlib.pyplot as plt
import lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import yaml
import wandb

from PIL import Image
from lightning.pytorch.loggers import WandbLogger
from torchvision.ops import sigmoid_focal_loss
from segmentation_models_pytorch.metrics import get_stats, iou_score, f1_score
from transformers import get_cosine_with_min_lr_schedule_with_warmup

from contrail_segmentation.data.plotting import plot_examples
from contrail_segmentation.data.utils import TEST_IDXS
from contrail_segmentation.train.utils import dice_coef
from contrail_segmentation.models.utils import compute_metrics

class PretrainedUNET(pl.LightningModule):
    
    def __init__(
        self, 
        encoder_class: nn.Module, 
        threshold: float = 0.5, 
        lr: float = 1e-3, 
        wd: float = 1e-3, 
        beta1: float = 0.9, 
        beta2: float = 0.999, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
    
        self.lr = lr
        self.wd = wd
        self.betas = (beta1, beta2)
        
        self.model = encoder_class()
        self.threshold = threshold
        self.sigmoid = nn.Sigmoid()
        
        self.bce_loss = smp.losses.FocalLoss(mode='binary')
        self.dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
        
    def _forward_pass(self, batch):
        imgs, targets = batch 
        y_hat = self.model(imgs)
        loss = self.bce_loss(y_hat, targets) + self.dice_loss(y_hat, targets)
        dice = dice_coef(targets, y_hat.detach(), thr=self.threshold)
        metrics = compute_metrics(y_hat.detach(), targets, thr=self.threshold)
        metrics['dice'] = dice
        
        return loss, metrics
    
    def training_step(self, batch, batch_idx):
        loss, metrics = self._forward_pass(batch)
        
        self.log(
            'train/loss', 
            loss, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True
        )
        
        
        for metric, value in metrics.items():
            self.log(
                f'train/{metric}', 
                value, 
                on_step=False, 
                on_epoch=True, 
                prog_bar=True
            )
        
    
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, metrics = self._forward_pass(batch)
        
        self.log(
            'val/loss', 
            loss, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True
        )
        
        for metric, value in metrics.items():
            self.log(
                f'val/{metric}', 
                value, 
                on_step=False, 
                on_epoch=True, 
                prog_bar=True
            )
    
        return loss 
    
    def test_step(self, batch, batch_idx):
        imgs, targets = batch
        y_hat = self.model(imgs)
        loss = self.bce_loss(y_hat, targets) + self.dice_loss(y_hat, targets)
        y_pred = self.sigmoid(y_hat)
        dice_loss = dice_coef(targets, y_pred, thr=self.threshold)
        metrics = compute_metrics(y_hat.detach(), targets, thr=self.threshold)
        metrics['dice'] = dice_loss
        
        self.log(
            'test/loss', 
            loss, 
            on_step=False, 
            on_epoch=True, 
            prog_bar=False
        )
        
        for metric, value in metrics.items():
            self.log(
                f'train/{metric}', 
                value, 
                on_step=False, 
                on_epoch=True, 
                prog_bar=True
            )
        
        return loss 
    
    def on_test_epoch_end(self):
        fig, axes = plot_examples(self, idxs=TEST_IDXS, mask_only=self.mask_only)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        
        self.logger.experiment.log({'Validation Examples': wandb.Image(img)})
        plt.close(fig)
        
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                     weight_decay=self.wd, 
                                     betas=self.betas)
        
        total_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(0.05 * total_steps)
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, 
                                                    num_training_steps=total_steps, min_lr=5e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "step",    
                "frequency": 1,         
            },
        }
    