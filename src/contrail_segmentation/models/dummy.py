import io 
import matplotlib.pyplot as plt
import lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import yaml
import wandb

from PIL import Image
from contrail_segmentation.data.plotting import plot_examples
from contrail_segmentation.data.utils import TEST_IDXS
from contrail_segmentation.train.utils import dice_coef
from transformers import get_cosine_with_min_lr_schedule_with_warmup

class Dummy(pl.LightningModule):
    
    def __init__(
        self,
        ones: bool = False,  
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ones = ones
        
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.bce_loss = smp.losses.FocalLoss(mode='binary')
        self.dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
        self.threshold = 0.5
        self.sigmoid = nn.Sigmoid()
        
    def model(self, x):
        shape = (x.size(0), 1, x.size(2), x.size(3))
        return torch.ones(shape, device=x.device) if self.ones else torch.zeros(shape, device=x.device)
        
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
        # redundant
        optimizer = torch.optim.Adam(self.parameters())
        return {
            "optimizer": optimizer,
        }