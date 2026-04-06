import lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

from torchvision.transforms.functional import gaussian_blur

from contrail_segmentation.train.utils import dice_coef
from contrail_segmentation.models.utils import compute_metrics
from contrail_segmentation.models.losses import SRLoss

# class PatchMasker(nn.Module):
#     def __init__(
#         self, 
#         input_dim: int = 3, 
#         patch_size: int = 16, 
#         hidden_channels: list = [32, 64],
#         kernel_sizes: list = [3, 3],
#         tau: float = 1.0,
#         *args, 
#         **kwargs
#     ):
#         super().__init__()
#         self.P = patch_size
#         self.tau = tau
        
#         assert len(hidden_channels) == len(kernel_sizes)
        
#         self.patch_net = nn.Sequential()
#         self.patch_net.append(nn.Conv2d(input_dim, hidden_channels[0], kernel_size=kernel_sizes[0], padding='same'))
#         self.patch_net.append(nn.ReLU())
        
#         for i in range(len(hidden_channels) - 1):
#             self.patch_net.append(nn.Conv2d(hidden_channels[i], hidden_channels[i+1], kernel_size=kernel_sizes[i+1], padding='same'))
#             self.patch_net.append(nn.ReLU())

#         self.patch_net.append(nn.AdaptiveAvgPool2d(1))
#         self.patch_net.append(nn.Flatten())
#         self.patch_net.append(nn.Linear(hidden_channels[-1], 2))

#     def forward(self, x: torch.Tensor):
#         B, C, H, W = x.shape
#         patches = x.unfold(2, self.P, self.P).unfold(3, self.P, self.P)
#         B, C, nH, nW, P, P = patches.shape
#         patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, P, P)
        
#         logits = self.patch_net(patches)
#         gumbel_out = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
#         patch_probs = gumbel_out[:, 1]
        
#         mask_small = patch_probs.view(B, 1, nH, nW)
#         mask_full = mask_small.repeat_interleave(self.P, dim=2).repeat_interleave(self.P, dim=3)
#         return mask_full

# class Masker(nn.Module):
#     def __init__(
#             self, 
#             input_dim: int = 3, 
#             kernel_sizes: list = [15, 7, 11], 
#             groups: list = [1, 1],
#             hidden_channels: list = [32, 64],
#             tau: float = 1.0,
#             *args, 
#             **kwargs
#         ):
#         super().__init__(*args, **kwargs)
#         self.tau = tau
#         self.layers = nn.Sequential()
        
#         self.layers.append(nn.Conv2d(input_dim, hidden_channels[0], kernel_size=kernel_sizes[0], padding='same', groups=groups[0]))
#         self.layers.append(nn.ReLU())
        
#         for i in range(len(hidden_channels) - 1):
#             self.layers.append(nn.Conv2d(hidden_channels[i], hidden_channels[i+1], kernel_size=kernel_sizes[i+1], padding='same', groups=groups[i+1]))
#             self.layers.append(nn.ReLU())

#         self.layers.append(nn.Conv2d(hidden_channels[-1], 2, kernel_size=kernel_sizes[-1], padding='same'))

#     def forward(self, x: torch.Tensor):
#         logits = self.layers(x)
#         logits = logits.permute(0, 2, 3, 1)
#         gumbel_out = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)
#         mask = gumbel_out[..., 1].unsqueeze(1)
#         return mask
class PatchMasker(nn.Module):
    def __init__(
        self, 
        input_dim: int = 3, 
        patch_size: int = 16, 
        hidden_channels: list = [32, 64],
        kernel_sizes: list = [3, 3],
        *args, 
        **kwargs
    ):
        super().__init__()
        self.P = patch_size
        
        assert len(hidden_channels) == len(kernel_sizes)
        
        self.patch_net = nn.Sequential()
        
        self.patch_net.append(nn.Conv2d(
            input_dim, 
            hidden_channels[0], 
            kernel_size=kernel_sizes[0],
            padding='same'
        ))
        self.patch_net.append(nn.ReLU())
        
        for i in range(len(hidden_channels) - 1):
            self.patch_net.append(nn.Conv2d(
                hidden_channels[i], 
                hidden_channels[i+1], 
                kernel_size=kernel_sizes[i+1],
                padding='same'
            ))
            self.patch_net.append(nn.ReLU())

        self.patch_net.append(nn.AdaptiveAvgPool2d(1))
        self.patch_net.append(nn.Flatten())
        self.patch_net.append(nn.Linear(hidden_channels[-1], 1))

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        
        patches = x.unfold(2, self.P, self.P).unfold(3, self.P, self.P)
        B, C, nH, nW, P, P = patches.shape
        
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, P, P)
        
        patch_probs = torch.sigmoid(self.patch_net(patches)) 
        
        mask_small = patch_probs.view(B, 1, nH, nW)
        mask_full = mask_small.repeat_interleave(self.P, dim=2).repeat_interleave(self.P, dim=3)
        
        return mask_full

class Masker(nn.Module):
    def __init__(
            self, 
            input_dim: int = 3, 
            kernel_sizes: list = list([15, 7, 11]), 
            groups: list = [1, 1],
            hidden_channels: list = [32, 64],
            *args, 
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        assert len(groups) == len(hidden_channels), "Groups list must match hidden_channels length"
 
        self.layers = nn.Sequential()
        
        self.layers.append(nn.Conv2d(
            input_dim, 
            hidden_channels[0], 
            kernel_size=kernel_sizes[0],
            padding='same',
            groups=groups[0], 
        ))
        self.layers.append(nn.ReLU())
        
        for i in range(len(hidden_channels) - 1):
            dim1 = hidden_channels[i]
            dim2 = hidden_channels[i+1]
            current_group = groups[i+1]
            kernel_size = kernel_sizes[i+1] 
            
            self.layers.append(nn.Conv2d(
                dim1, 
                dim2, 
                kernel_size=kernel_size,
                padding='same',
                groups=current_group,
            ))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Conv2d(
            hidden_channels[-1], 
            1,
            kernel_size=kernel_sizes[-1], 
            padding='same'
        ))

    def forward(self, x: torch.Tensor):
        out = self.layers(x)
        return out

class MaskedUNET(pl.LightningModule):
    
    def __init__(
        self, 
        soft: bool, 
        encoder_class: nn.Module, 
        threshold: float = 0.5, 
        masker: nn.Module = Masker, 
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
        self.save_hyperparameters()
    
        self.lr = lr
        self.wd = wd
        self.betas = (beta1, beta2)
        
        self.model = encoder_class()
        self.masker = masker()
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
        
    def entropy_loss(self, mask, eps=1e-8):
        entropy = -(mask * torch.log(mask + eps) + (1 - mask) * torch.log(1 - mask + eps))
        return entropy.mean()
    
    def plateau_cloud(self, mask, dilation_size=35, blur_sigma=15):
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
            
        padding = dilation_size // 2
        thick_mask = F.max_pool2d(mask.float(), kernel_size=dilation_size, stride=1, padding=padding)
        
        k_size = int(math.ceil(4 * blur_sigma)) | 1 
        soft_targets = gaussian_blur(thick_mask, [k_size, k_size], sigma=[blur_sigma, blur_sigma])
        
        view_shape = soft_targets.shape
        flat_targets = soft_targets.view(view_shape[0], -1)
        max_per_sample = flat_targets.max(dim=1, keepdim=True)[0]
        
        soft_targets = flat_targets / (max_per_sample + 1e-8)
        
        return soft_targets.view(view_shape) 

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
        
        clouds = self.plateau_cloud(targets).detach()
        mask_loss = F.binary_cross_entropy_with_logits(masks, clouds)
        loss = loss + self.sparse_weight * mask_loss

        return loss, mask_loss, hough_loss      
    
    def forward(self, x, return_mask=False):
        mask = self.masker(x)
        mask = F.sigmoid(mask)
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
        masks = F.sigmoid(masks)
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