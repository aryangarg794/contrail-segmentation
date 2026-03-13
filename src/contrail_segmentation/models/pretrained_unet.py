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

from contrail_segmentation.data.plotting import plot_val_examples
from contrail_segmentation.train.utils import dice_coef

class PretrainedUNET(pl.LightningModule):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        with open('src/contrail_segmentation/config/models/pretrained_unet.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
            file.close()
            
        with open('src/contrail_segmentation/config/optim/adam.yaml', 'r') as file:
            self.opt_config = yaml.safe_load(file)
            file.close()
            
        self.model = smp.Unet(**self.config['model_params'])
        self.threshold = self.config['threshold']
        self.sigmoid = nn.Sigmoid()
        
        self.bce_loss = smp.losses.SoftBCEWithLogitsLoss(pos_weight=torch.tensor([100.0]))
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
        fig, axes = plot_val_examples(self)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        
        self.logger.experiment.log({'Validation Examples': wandb.Image(img)})
        plt.close(fig)
        
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt_config['lr'],
                                     weight_decay=self.opt_config['weight_decay'], 
                                     betas=(self.opt_config['beta1'], self.opt_config['beta2']))
        
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
    