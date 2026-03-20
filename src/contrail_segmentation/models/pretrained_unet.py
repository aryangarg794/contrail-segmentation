# import io 
# import matplotlib.pyplot as plt
# import lightning as pl
# import segmentation_models_pytorch as smp
# import torch
# import torch.nn as nn
# import yaml
# import wandb

# from PIL import Image
# from lightning.pytorch.loggers import WandbLogger
# from torchvision.ops import sigmoid_focal_loss
# from transformers import get_cosine_schedule_with_warmup

# from contrail_segmentation.data.plotting import plot_val_examples
# from contrail_segmentation.train.utils import dice_coef

# class PretrainedUNET(pl.LightningModule):
    
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
        
#         with open('src/contrail_segmentation/config/models/pretrained_unet.yaml', 'r') as file:
#             self.config = yaml.safe_load(file)
#             file.close()
            
#         with open('src/contrail_segmentation/config/optim/adam.yaml', 'r') as file:
#             self.opt_config = yaml.safe_load(file)
#             file.close()
            
#         self.model = smp.Unet(**self.config['model_params'])
#         self.threshold = self.config['threshold']
#         self.sigmoid = nn.Sigmoid()
        
#         self.bce_loss = smp.losses.SoftBCEWithLogitsLoss(pos_weight=torch.tensor([100.0]))
#         self.dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
        
#     def _forward_pass(self, batch):
#         imgs, targets = batch 
#         y_hat = self.model(imgs)
#         loss = self.bce_loss(y_hat, targets) + self.dice_loss(y_hat, targets)
#         dice = dice_coef(targets, y_hat.detach(), thr=self.threshold)
        
#         return loss, dice
    
#     def training_step(self, batch, batch_idx):
#         loss, dice = self._forward_pass(batch)
        
#         self.log(
#             'train/loss', 
#             loss, 
#             on_step=True, 
#             on_epoch=True, 
#             prog_bar=True
#         )
        
#         self.log(
#             'train/dice', 
#             dice, 
#             on_step=False, 
#             on_epoch=True, 
#             prog_bar=True
#         )
    
#         return loss
    
#     def validation_step(self, batch, batch_idx, dataloader_idx=0):
#         loss, dice = self._forward_pass(batch)
        
#         self.log(
#             'val/loss', 
#             loss, 
#             on_step=True, 
#             on_epoch=True, 
#             prog_bar=True
#         )
        
#         self.log(
#             'val/dice', 
#             dice, 
#             on_step=False, 
#             on_epoch=True, 
#             prog_bar=True
#         )
    
#         return loss 
    
#     def test_step(self, batch, batch_idx):
#         imgs, targets = batch
#         y_hat = self.model(imgs)
#         loss = self.bce_loss(y_hat, targets) + self.dice_loss(y_hat, targets)
#         y_pred = self.sigmoid(y_hat)
#         dice_loss = dice_coef(targets, y_pred, thr=self.threshold)
        
#         self.log(
#             'test/loss', 
#             loss, 
#             on_step=False, 
#             on_epoch=True, 
#             prog_bar=False
#         )
        
#         self.log(
#             'test/dice', 
#             dice_loss, 
#             on_step=False, 
#             on_epoch=True, 
#             prog_bar=False
#         )
        
#         return loss 
    
#     def on_test_epoch_end(self):
#         fig, axes = plot_val_examples(self)
#         buf = io.BytesIO()
#         fig.savefig(buf, format='png')
#         buf.seek(0)
#         img = Image.open(buf)
        
#         self.logger.experiment.log({'Validation Examples': wandb.Image(img)})
#         plt.close(fig)
        
    
#     def configure_optimizers(self):
        
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=self.opt_config['lr'],
#                                      weight_decay=self.opt_config['weight_decay'], 
#                                      betas=(self.opt_config['beta1'], self.opt_config['beta2']))
        
#         total_steps = self.trainer.estimated_stepping_batches
#         num_warmup_steps = int(0.05 * total_steps)
#         scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, 
#                                                     num_training_steps=total_steps)
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler, 
#                 "interval": "step",    
#                 "frequency": 1,         
#             },
#         }
    




































import io
import math
import matplotlib.pyplot as plt
import lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import yaml
import wandb

from PIL import Image
from transformers import get_cosine_schedule_with_warmup

from contrail_segmentation.data.plotting import plot_val_examples
from contrail_segmentation.train.utils import dice_coef


class HoughSRLoss(nn.Module):
    """
    SR-style loss:
      - Dice loss in image space
      - Dice loss in a Hough-like accumulator space

    This is a usable PyTorch approximation of the paper idea.
    """

    def __init__(self, alpha=0.5, num_theta=180, rho_bins=512, line_thresh=50):
        super().__init__()
        self.alpha = alpha
        self.num_theta = num_theta
        self.rho_bins = rho_bins
        self.line_thresh = line_thresh
        self.dice_prob = smp.losses.DiceLoss(mode="binary", from_logits=False)

    def single_hough_map(self, mask2d: torch.Tensor) -> torch.Tensor:
        """
        mask2d: H x W, values in [0,1]
        returns: rho_bins x num_theta accumulator
        """
        device = mask2d.device
        H, W = mask2d.shape

        binary = (mask2d > 0.5).float()

        ys, xs = torch.where(binary > 0)
        if xs.numel() == 0:
            return torch.zeros((self.rho_bins, self.num_theta), device=device)

        thetas = torch.linspace(-math.pi / 2, math.pi / 2, self.num_theta, device=device)
        cos_t = torch.cos(thetas)
        sin_t = torch.sin(thetas)

        diag = math.sqrt(H * H + W * W)
        rho_min, rho_max = -diag, diag

        acc = torch.zeros((self.rho_bins, self.num_theta), device=device)

        xs = xs.float()
        ys = ys.float()

        for t_idx in range(self.num_theta):
            rho_vals = xs * cos_t[t_idx] + ys * sin_t[t_idx]
            rho_idx = ((rho_vals - rho_min) / (rho_max - rho_min) * (self.rho_bins - 1)).long()
            rho_idx = torch.clamp(rho_idx, 0, self.rho_bins - 1)

            counts = torch.bincount(rho_idx, minlength=self.rho_bins).float()
            acc[:, t_idx] = counts

        acc = torch.where(acc >= self.line_thresh, acc, torch.zeros_like(acc))

        if acc.max() > 0:
            acc = acc / acc.max()

        return acc

    def batch_hough_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: B x 1 x H x W
        returns: B x 1 x rho_bins x num_theta
        """
        maps = []
        for i in range(x.shape[0]):
            maps.append(self.single_hough_map(x[i, 0]))
        return torch.stack(maps, dim=0).unsqueeze(1)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        loss_img = self.dice_prob(probs, targets)

        pred_h = self.batch_hough_maps(probs)
        targ_h = self.batch_hough_maps(targets)

        loss_h = self.dice_prob(pred_h, targ_h)

        return (1.0 - self.alpha) * loss_img + self.alpha * loss_h


class PretrainedUNET(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        with open("src/contrail_segmentation/config/models/pretrained_unet.yaml", "r") as file:
            self.config = yaml.safe_load(file)

        with open("src/contrail_segmentation/config/optim/adam.yaml", "r") as file:
            self.opt_config = yaml.safe_load(file)

        self.model = smp.Unet(**self.config["model_params"])
        self.threshold = self.config["threshold"]
        self.sigmoid = nn.Sigmoid()

        self.bce_loss = smp.losses.SoftBCEWithLogitsLoss(
            pos_weight=torch.tensor([100.0])
        )
        self.dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)

        # New Hough-based SR loss
        self.sr_loss = HoughSRLoss(alpha=0.5, num_theta=180, rho_bins=512, line_thresh=50)

    def _forward_pass(self, batch):
        imgs, targets = batch
        y_hat = self.model(imgs)

        # Baseline + SR-style Hough loss
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

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/dice", dice_score, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def on_test_epoch_end(self):
        fig, axes = plot_val_examples(self)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img = Image.open(buf)

        self.logger.experiment.log({"Validation Examples": wandb.Image(img)})
        plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.opt_config["lr"],
            weight_decay=self.opt_config["weight_decay"],
            betas=(self.opt_config["beta1"], self.opt_config["beta2"]),
        )

        total_steps = self.trainer.estimated_stepping_batches
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