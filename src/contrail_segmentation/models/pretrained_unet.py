import io 
import matplotlib.pyplot as plt
import lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import yaml
import wandb

from PIL import Image
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
        dice_weight: float = 0.5, 
        focal_weight: float = 0.5,
        bce_loss: nn.Module = nn.BCEWithLogitsLoss, 
        dice_loss: nn.Module = smp.losses.DiceLoss,
        pos_weight: int = 10, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
    
        self.lr = lr
        self.wd = wd
        self.betas = (beta1, beta2)
        
        self.model = encoder_class()
        self.encoder_name = encoder_class.keywords.get("encoder_name")
        self.encoder_weights = encoder_class.keywords.get("encoder_weights")
        
        self.threshold = threshold
        
        if isinstance(bce_loss, nn.BCEWithLogitsLoss):
            pos_weight = torch.tensor([pos_weight])
            self.bce_loss = bce_loss(pos_weight=pos_weight)
        else: 
            self.bce_loss = bce_loss()

        self.dice_loss = dice_loss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
    def _loss(self, preds, targets):
        return self.dice_weight * self.dice_loss(preds, targets) + \
            self.focal_weight * self.bce_loss(preds, targets)              
        
    def _forward_pass(self, batch):
        imgs, targets = batch 
        y_hat = self.model(imgs)
        loss = self._loss(y_hat, targets)
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
                prog_bar=True if metric == 'dice' else False
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
        loss = self._loss(y_hat, targets)
        dice_loss = dice_coef(targets, y_hat.detach(), thr=self.threshold)
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
                f'test/{metric}', 
                value, 
                on_step=False, 
                on_epoch=True, 
                prog_bar=True
            )
        
        return loss 
    
    def on_test_epoch_end(self):
        self.log('test/threshold', self.threshold, prog_bar=False, on_epoch=True, on_step=False)
        fig, axes = plot_examples(self, idxs=TEST_IDXS, mask_only=self.mask_only)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        
        self.logger.experiment.log({'Validation Examples': wandb.Image(img)})
        plt.close(fig)
        
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr,
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
    