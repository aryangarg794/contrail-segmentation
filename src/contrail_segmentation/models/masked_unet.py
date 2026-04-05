import lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

from contrail_segmentation.train.utils import dice_coef
from contrail_segmentation.models.utils import compute_metrics
from contrail_segmentation.models.losses import SRLoss

class Masker(nn.Module):

    def __init__(
            self, 
            input_dim: int = 3, 
            kernel_size: int = 7, 
            hidden_channels: list = list([32, 64]),
            *args, 
            **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.layers = nn.Sequential()
        self.layers.extend([nn.Conv2d(
            input_dim, 
            hidden_channels[0], 
            kernel_size=kernel_size,
            padding='same'
        ), nn.ReLU()])
        
        for dim1, dim2 in zip(hidden_channels[:-1], hidden_channels[1:]):
            self.layers.extend([nn.Conv2d(
                dim1, 
                dim2, 
                kernel_size=kernel_size,
                padding='same'
            ), nn.ReLU()])

        self.layers.append(nn.Conv2d(
            hidden_channels[-1], 
            1,
            kernel_size=kernel_size, 
            padding='same'
        ))


    def forward(self, x: torch.Tensor):
        return F.sigmoid(self.layers(x))

class MaskedUNET(pl.LightningModule):
    
    def __init__(
        self, 
        soft: bool, 
        encoder_class: nn.Module, 
        threshold: float = 0.5, 
        masker_kernel: int = 7, 
        masker_channels: list = list([32, 64]), 
        lr: float = 1e-3, 
        wd: float = 1e-3, 
        beta1: float = 0.9, 
        beta2: float = 0.999, 
        hough: bool = False, 
        dice_weight: float = 0.5, 
        focal_weight: float = 0.5,
        sparse_weight: float = 0.01, 
        bce_loss: nn.Module = nn.BCEWithLogitsLoss, 
        dice_loss: nn.Module = smp.losses.DiceLoss,
        pos_weight: int = 1, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
    
        self.lr = lr
        self.wd = wd
        self.betas = (beta1, beta2)
        
        self.model = encoder_class()
        self.masker = Masker(kernel_size=masker_kernel, hidden_channels=masker_channels)
        self.encoder_name = encoder_class.keywords.get("encoder_name")
        self.encoder_weights = encoder_class.keywords.get("encoder_weights")
        
        self.threshold = threshold
        self.soft = soft
        
        if isinstance(bce_loss, nn.BCEWithLogitsLoss):
            pos_weight = torch.tensor([pos_weight])
            self.bce_loss = bce_loss(pos_weight=pos_weight)
        else: 
            self.bce_loss = bce_loss()

        if hough:
            self.sr_loss = SRLoss()
        self.hough = hough
        self.hough_weight = 1/3
        
        self.dice_loss = dice_loss()
        if hough:
            self.dice_weight = 1/3
            self.focal_weight = 1/3
        else:
            self.dice_weight = dice_weight
            self.focal_weight = focal_weight
        self.sparse_weight = sparse_weight
        
    def _loss(self, preds, targets, masks, targets_soft=None):
        if self.soft:
            loss = self.dice_weight * self.dice_loss(preds, targets_soft) + \
            self.focal_weight * self.bce_loss(preds, targets)  
        else:
            loss = self.dice_weight * self.dice_loss(preds, targets) + \
            self.focal_weight * self.bce_loss(preds, targets)

        hough_loss = None
        if self.hough:
            hough_loss = self.sr_loss(preds, targets)
            loss = loss + self.hough_weight * hough_loss

        sparsity_loss = masks.squeeze(dim=-1).mean(dim=(1, 2)).mean()
        loss = loss + self.sparse_weight * sparsity_loss
        return loss, sparsity_loss, hough_loss      
    
    def forward(self, x, return_mask=False):
        mask = self.masker(x)
        if return_mask:
            return self.model(x * mask), mask
        else: 
            return self.model(x * mask)
        
    def _forward_pass(self, batch):
        if self.soft: 
            imgs, targets, target_softs = batch
        else:
            imgs, targets = batch
            target_softs = None

        masks = self.masker(imgs)
        y_hat = self.model(imgs * masks)
        
        loss, sparse_loss, hough_loss = self._loss(y_hat, targets, masks, target_softs)
        dice = dice_coef(targets, y_hat.detach(), thr=self.threshold)
        metrics = compute_metrics(y_hat.detach(), targets, thr=self.threshold)
        metrics['dice'] = dice
        metrics['sparse_loss'] = sparse_loss
        if hough_loss:
            metrics['hough_loss'] = hough_loss
        
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
        loss, metrics = self._forward_pass(batch)
        
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
        
    
    def configure_optimizers(self):

        params = list(self.model.parameters()) + list(self.masker.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.lr,
                                     weight_decay=self.wd, 
                                     betas=self.betas)
        
        total_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(0.05 * total_steps)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-2, end_factor=1.0, total_iters=num_warmup_steps,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - num_warmup_steps, eta_min=1e-7,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[num_warmup_steps],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }