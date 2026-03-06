import lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import yaml

from lightning.pytorch.loggers import WandbLogger
from torchvision.ops import sigmoid_focal_loss
from segmentation_models_pytorch.losses import DiceLoss

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
        self.dice = DiceLoss('binary', from_logits='False')
        
    def _forward_pass(self, batch):
        imgs, targets = batch 
        y_hat = self.model(imgs)
        loss = sigmoid_focal_loss(y_hat, targets, alpha=0.5, reduction="mean")
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self._forward_pass(batch)
        self.log(
            'train/loss', 
            loss, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True
        )
    
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self._forward_pass(batch)
        self.log(
            'val/loss',
            loss,  
            on_step=False, 
            on_epoch=True, 
            prog_bar=False
        )
    
        return loss 
    
    def test_step(self, batch, batch_idx):
        imgs, targets = batch
        y_hat = self.model(imgs)
        loss = sigmoid_focal_loss(y_hat, targets, alpha=0.5, reduction="mean")
        y_thresh = (self.sigmoid(y_hat) >= self.threshold).float() 
        dice_loss = self.dice(y_thresh, targets)
        
        self.log(
            'test/loss', 
            loss, 
            on_step=False, 
            on_epoch=True, 
            prog_bar=False
        )
        
        self.log(
            'test/dice_score', 
            dice_loss, 
            on_step=False, 
            on_epoch=True, 
            prog_bar=False
        )
        
        return loss 
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt_config['lr'],
                                     weight_decay=self.opt_config['weight_decay'], 
                                     betas=(self.opt_config['beta1'], self.opt_config['beta2']))
        return optimizer