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
from transformers import get_cosine_schedule_with_warmup

from contrail_segmentation.data.plotting import plot_examples
from contrail_segmentation.data.utils import TEST_IDXS
from contrail_segmentation.train.utils import dice_coef

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
        
        return loss, dice
    
    def training_step(self, batch, batch_idx):
        loss, dice = self._forward_pass(batch)
        
        self.log(
            'train/loss', 
            loss, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True
        )
        
        self.log(
            'train/dice', 
            dice, 
            on_step=False, 
            on_epoch=True, 
            prog_bar=True
        )
    
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, dice = self._forward_pass(batch)
        
        self.log(
            'val/loss', 
            loss, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True
        )
        
        self.log(
            'val/dice', 
            dice, 
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
        
        self.log(
            'test/loss', 
            loss, 
            on_step=False, 
            on_epoch=True, 
            prog_bar=False
        )
        
        self.log(
            'test/dice', 
            dice_loss, 
            on_step=False, 
            on_epoch=True, 
            prog_bar=False
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
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, 
                                                    num_training_steps=total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "step",    
                "frequency": 1,         
            },
        }
    