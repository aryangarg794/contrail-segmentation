import io
import math
import yaml
import wandb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import lightning as pl
import segmentation_models_pytorch as smp

from PIL import Image
from transformers import get_cosine_schedule_with_warmup
from contrail_segmentation.data.plotting import plot_val_examples
from contrail_segmentation.train.utils import dice_coef

class HoughSRLoss(nn.Module):
    """
    Differentiable Shape-Regularized (SR) Loss using a Soft Hough Transform.
    Encodes the 'lininess' inductive bias for contrail segmentation.
    """
    def __init__(self, alpha=0.5, num_theta=90, rho_bins=256, line_thresh=50):
        super().__init__()
        self.alpha = alpha
        self.num_theta = num_theta
        self.rho_bins = rho_bins
        self.line_thresh = line_thresh
        self.dice_prob = smp.losses.DiceLoss(mode="binary", from_logits=False)

    def single_hough_map(self, mask2d: torch.Tensor) -> torch.Tensor:
        """
        Calculates a differentiable Hough/Radon map from 2D sigmoid probabilities.
        """
        device = mask2d.device
        H, W = mask2d.shape

        # 1. Create coordinate grid
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )

        # 2. Vectorized Rho calculation
        thetas = torch.linspace(-math.pi / 2, math.pi / 2, self.num_theta, device=device)
        cos_t = torch.cos(thetas).view(-1, 1, 1)
        sin_t = torch.sin(thetas).view(-1, 1, 1)
        
        rho = x * cos_t + y * sin_t

        # 3. Binning logic (Linear mapping to indices)
        rho_idx = ((rho + 1.414) / 2.828 * (self.rho_bins - 1)).long()
        rho_idx = torch.clamp(rho_idx, 0, self.rho_bins - 1)

        # 4. Soft Accumulation using scatter_add_
        acc = torch.zeros((self.num_theta, self.rho_bins), device=device)
        mask_flat = mask2d.reshape(-1)

        for t in range(self.num_theta):
            acc[t].scatter_add_(0, rho_idx[t].reshape(-1), mask_flat)

        # 5. Differentiable Line Thresholding
        acc = torch.relu(acc - self.line_thresh)

        # 6. Peak Normalization
        if acc.max() > 0:
            acc = acc / (acc.max() + 1e-6)

        return acc.t() # Output shape: (rho_bins, num_theta)

    def batch_hough_maps(self, x: torch.Tensor) -> torch.Tensor:
        maps = [self.single_hough_map(x[i, 0]) for i in range(x.shape[0])]
        return torch.stack(maps, dim=0).unsqueeze(1)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Image-space loss
        loss_img = self.dice_prob(probs, targets)

        # Hough-space loss (MSE between accumulators)
        pred_h = self.batch_hough_maps(probs)
        targ_h = self.batch_hough_maps(targets)
        loss_h = torch.nn.functional.mse_loss(pred_h, targ_h)

        return (1.0 - self.alpha) * loss_img + self.alpha * loss_h


class PretrainedUNET(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load configurations
        with open("src/contrail_segmentation/config/models/pretrained_unet.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        with open("src/contrail_segmentation/config/optim/adam.yaml", "r") as f:
            self.opt_config = yaml.safe_load(f)

        # Model setup
        self.model = smp.Unet(**self.config["model_params"])
        self.threshold = self.config["threshold"]
        self.sigmoid = nn.Sigmoid()

        # Loss definitions
        self.bce_loss = smp.losses.SoftBCEWithLogitsLoss(pos_weight=torch.tensor([100.0]))
        self.sr_loss = HoughSRLoss(alpha=0.5, num_theta=90, rho_bins=256, line_thresh=50)

    def _forward_pass(self, batch):
        imgs, targets = batch
        y_hat = self.model(imgs)

        # Combined Pixel-wise and Geometric Loss
        loss = self.bce_loss(y_hat, targets) + self.sr_loss(y_hat, targets)
        dice = dice_coef(targets, y_hat.detach(), thr=self.threshold)
        
        return loss, dice

    def training_step(self, batch, batch_idx):
        loss, dice = self._forward_pass(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/dice", dice, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, dice = self._forward_pass(batch)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/dice", dice, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        imgs, targets = batch
        y_hat = self.model(imgs)
        loss = self.bce_loss(y_hat, targets) + self.sr_loss(y_hat, targets)
        
        y_pred = self.sigmoid(y_hat)
        dice_score = dice_coef(targets, y_pred, thr=self.threshold)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/dice", dice_score, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        fig, _ = plot_val_examples(self)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        self.logger.experiment.log({"Validation Examples": wandb.Image(Image.open(buf))})
        plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.opt_config["lr"],
            weight_decay=self.opt_config["weight_decay"],
            betas=(self.opt_config["beta1"], self.opt_config["beta2"]),
        )

        total_steps = self.trainer.estimated_stepping_batches
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.05 * total_steps),
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
