import io
import math
import matplotlib.pyplot as plt
import lightning as pl
import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from PIL import Image
from transformers import get_cosine_schedule_with_warmup

from contrail_segmentation.data.plotting import plot_examples
from contrail_segmentation.data.utils import TEST_IDXS
from contrail_segmentation.train.losses import SRLoss
from contrail_segmentation.train.utils import dice_coef

class ConvNeXtBlock(nn.Module):

    def __init__(self, dim: int, expansion: int = 4, kernel_size: int = 7):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                                padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, expansion * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return x + residual


class AttentionGate(nn.Module):

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        psi = self.relu(self.W_g(g) + self.W_x(x))
        return x * self.psi(psi)

class AttentionUNetConvNeXtModel(nn.Module):
 
    def __init__(
        self,
        in_channels: int = 24,
        backbone: str = "convnext_base",
    ):
        super().__init__()

        # Backbone encoder (fully trainable)
        self.backbone = timm.create_model(
            backbone, pretrained=False, features_only=True, in_chans=in_channels,
        )
        f1_ch, f2_ch, f3_ch, f4_ch = [s['num_chs'] for s in self.backbone.feature_info]

        # Bottleneck
        self.bottleneck = ConvNeXtBlock(f4_ch)

        # Decoder
        self.up3 = nn.ConvTranspose2d(f4_ch, f3_ch, kernel_size=2, stride=2)
        self.attn3 = AttentionGate(F_g=f3_ch, F_l=f3_ch, F_int=f3_ch // 2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(f3_ch * 2, f3_ch, kernel_size=1, bias=False),
            ConvNeXtBlock(f3_ch),
        )

        self.up2 = nn.ConvTranspose2d(f3_ch, f2_ch, kernel_size=2, stride=2)
        self.attn2 = AttentionGate(F_g=f2_ch, F_l=f2_ch, F_int=f2_ch // 2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(f2_ch * 2, f2_ch, kernel_size=1, bias=False),
            ConvNeXtBlock(f2_ch),
        )

        self.up1 = nn.ConvTranspose2d(f2_ch, f1_ch, kernel_size=2, stride=2)
        self.attn1 = AttentionGate(F_g=f1_ch, F_l=f1_ch, F_int=f1_ch // 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(f1_ch * 2, f1_ch, kernel_size=1, bias=False),
            ConvNeXtBlock(f1_ch),
        )

        # Head: 4x bilinear upsample to full resolution, then classify
        self.head = nn.Sequential(
            nn.Conv2d(f1_ch, f1_ch // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f1_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1_ch // 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2:]

        f1, f2, f3, f4 = self.backbone(x)

        b = self.bottleneck(f4)

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, self.attn3(d3, f3)], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self.attn2(d2, f2)], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self.attn1(d1, f1)], dim=1))

        d1 = F.interpolate(d1, size=(H, W), mode='bilinear', align_corners=False)
        return self.head(d1)


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class AttentionUNetConvNeXt(pl.LightningModule):

    def __init__(
        self,
        in_channels: int = 24,
        backbone: str = "convnext_base",
        threshold: float = 0.5,
        lr: float = 1e-4,
        wd: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        total_steps: int = 10000,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.lr = lr
        self.wd = wd
        self.betas = (beta1, beta2)
        self.threshold = threshold
        self.total_steps = total_steps

        self.model = AttentionUNetConvNeXtModel(
            in_channels=in_channels,
            backbone=backbone,
        )

        self.sigmoid = nn.Sigmoid()
        self.focal_loss = smp.losses.FocalLoss(mode='binary')
        self.sr_loss = SRLoss(H=256, W=256, num_angles=90, alpha=0.5)

    def _forward_pass(self, batch):
        imgs, targets = batch
        y_hat = self.model(imgs)
        loss = self.focal_loss(y_hat, targets) + self.sr_loss(y_hat, targets)
        dice = dice_coef(targets, y_hat.detach(), thr=self.threshold)
        return loss, dice

    def training_step(self, batch, batch_idx):
        loss, dice = self._forward_pass(batch)
        self.log('train/loss', loss, on_step=True,  on_epoch=True, prog_bar=True)
        self.log('train/dice', dice, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, dice = self._forward_pass(batch)
        self.log('val/loss', loss, on_step=True,  on_epoch=True, prog_bar=True)
        self.log('val/dice', dice, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        imgs, targets = batch
        y_hat = self.model(imgs)
        loss = self.focal_loss(y_hat, targets) + self.sr_loss(y_hat, targets)
        dice_loss = dice_coef(targets, y_hat, thr=self.threshold)
        self.log('test/loss', loss,      on_step=False, on_epoch=True, prog_bar=False)
        self.log('test/dice', dice_loss, on_step=False, on_epoch=True, prog_bar=False)
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
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.wd,
            betas=self.betas,
        )
        total_steps = self.total_steps
        num_warmup_steps = int(0.05 * total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
