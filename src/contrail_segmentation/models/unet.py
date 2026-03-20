import io 
import torch
import matplotlib.pyplot as plt
import numpy as np 
import lightning as pl
import segmentation_models_pytorch as smp
import torch.nn as nn
import yaml
import wandb

from PIL import Image
from torch import Tensor
from torchvision.ops import sigmoid_focal_loss
from transformers import get_cosine_with_min_lr_schedule_with_warmup
from typing import Self, List

from contrail_segmentation.data.plotting import plot_examples
from contrail_segmentation.data.utils import TEST_IDXS
from contrail_segmentation.train.utils import dice_coef
from contrail_segmentation.models.utils import compute_metrics
# implementation loosely inspired by https://medium.com/data-science/diffusion-model-from-scratch-in-pytorch-ddpm-9d9760528946

class Sinusoidal(nn.Module):
    
    def __init__(
        self: Self,
        embed_size: int, 
        horizon: int = 1000,
        *args, 
        **kwargs
    ) -> None:
        super(Sinusoidal, self).__init__(*args, **kwargs) 
        
        pe = torch.zeros(horizon, embed_size, requires_grad=False)
        positions = torch.arange(0, horizon).unsqueeze(dim=1)
        div = torch.exp(torch.arange(0, embed_size, 2).float() * -(np.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(positions * div)
        pe[:, 1::2] = torch.cos(positions * div)
        
        self.embed_size = embed_size
        self.register_buffer("pe", pe) 

    def forward(self: Self, t: int) -> Tensor:
        return self.pe[t].view(-1, self.embed_size, 1, 1)
    
    
class ResidualBlock(nn.Module):
    
    def __init__(
        self: Self,
        in_channels: int,
        dropout: float = 0.1, 
        activation: nn.Module = nn.ReLU,
        *args, 
        **kwargs
    ) -> None:
        super(ResidualBlock, self).__init__(*args, **kwargs)
        
        
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            activation(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(in_channels),
            activation(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        )
        
    def forward(self: Self, x: Tensor) -> Tensor:
        return self.layers(x) + x * 1/np.sqrt(2)
    
    
class UNETLayer(nn.Module):
    
    def __init__(
        self: Self,
        in_channels: int, 
        out_channels: int, 
        resblock: ResidualBlock = ResidualBlock,
        upsample: bool = False, 
        attention: bool = False, 
        num_heads: int = 8, 
        dropout: float = 0.1,
        *args, 
        **kwargs
    ) -> None:
        super(UNETLayer, self).__init__(*args, **kwargs)
        
        self.resblock1 = resblock(in_channels=in_channels)    
        self.resblock2 = resblock(in_channels=in_channels)
        
        if upsample:
            self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, 
                                           kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=3, stride=2, padding=1)
            
        self.attention = attention
        if attention: 
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=in_channels, 
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        
    def forward(self: Self, x: Tensor) -> None: 
        x = self.resblock1(x)
        if self.attention:
            batch_size, channels, height, width = x.shape
            x = x.view(batch_size, channels, -1).transpose(1, 2) # attention on patches
            x, _ = self.attention_layer(x, x, x)
            x = x.transpose(1, 2).view(batch_size, channels, height, width)
    
        x = self.resblock2(x)
        return self.conv(x), x
    
    
class UNETBase(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        enc_channels: List[int] = [32, 64, 128, 256, 512],
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
    
        self.init_conv = nn.Conv2d(in_channels, enc_channels[0], kernel_size=3, padding=1)
    
        self.enc1 = UNETLayer(enc_channels[0], enc_channels[1], upsample=False)
        self.enc2 = UNETLayer(enc_channels[1], enc_channels[2], upsample=False)
        self.enc3 = UNETLayer(enc_channels[2], enc_channels[3], upsample=False, attention=False)
        self.enc4 = UNETLayer(enc_channels[3], enc_channels[4], upsample=False, attention=False)
        
        self.up1 = UNETLayer(enc_channels[4], enc_channels[3], upsample=True, attention=False)
        self.up2 = UNETLayer(enc_channels[3] * 2, enc_channels[2], upsample=True, attention=False)
        self.up3 = UNETLayer(enc_channels[2] * 2, enc_channels[1], upsample=True)
        self.up4 = UNETLayer(enc_channels[1] * 2, enc_channels[0], upsample=True)
        
        final_in = enc_channels[0] * 2
        self.conv_final = nn.Sequential(
            nn.Conv2d(final_in, enc_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(enc_channels[0], out_channels, kernel_size=1),
            # nn.Conv2d(1, 1, kernel_size=2, padding=1, bias=False), # to fix the 0.5 shift            
        )

    def forward(self, x: Tensor) -> Tensor:
        s0 = self.init_conv(x) # channels[0]          
        x, s1 = self.enc1(s0)  # channels[1], channels[0]
        x, s2 = self.enc2(x) # channels[2], channels[1]          
        x, s3 = self.enc3(x) # channels[3], channels[2]      
        x, s4 = self.enc4(x) # channels[4], channels[3]     
        
        x, _ = self.up1(x) # channels[3]    
        # print(x.shape, s4.shape, s3.shape, s2.shape, s1.shape, s0.shape)         
        x, _ = self.up2(torch.cat([x, s4], dim=1)) # channels[2]
        x, _ = self.up3(torch.cat([x, s3], dim=1)) # channels[1]
        x, _ = self.up4(torch.cat([x, s2], dim=1)) # channels[0]
        
        x = self.conv_final(torch.cat([x, s0], dim=1)) 
        return x

        
class UNET(pl.LightningModule):
    
    def __init__(
        self, 
        threshold: float = 0.5,
        in_channels: int = 24,
        out_channels: int = 1,
        enc_channels: List[int] = [32, 64, 128, 256],
        lr: float = 1e-3, 
        wd: float = 1e-3, 
        beta1: float = 0.9, 
        beta2: float = 0.999,
        dice_weight: float = 0.5, 
        focal_weight: float = 0.5,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.lr = lr
        self.wd = wd
        self.betas = (beta1, beta2)
            
        self.model = UNETBase(in_channels=in_channels, out_channels=out_channels, enc_channels=enc_channels)
        self.threshold = threshold
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = smp.losses.FocalLoss(mode='binary')
        self.dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
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
                on_epoch=True if metric == 'dice' else False, 
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
    