from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.functional import dice


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs  = torch.sigmoid(logits)
        p_flat = probs.view(probs.size(0), -1)
        t_flat = targets.view(targets.size(0), -1).float()

        intersection = (p_flat * t_flat).sum(dim=1)
        score = (2.0 * intersection + self.smooth) / (
            p_flat.sum(dim=1) + t_flat.sum(dim=1) + self.smooth
        )
        return 1.0 - score.mean()


class PretrainedUNET(pl.LightningModule):
   
    //Baseline ResUNet lightning module

    def __init__(
        self,
        encoder_class: Callable[[], nn.Module],
        lr:        float = 1e-3,
        wd:        float = 1e-4,
        beta1:     float = 0.9,
        beta2:     float = 0.999,
        threshold: float = 0.5,
        alpha:     float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder_class"])

        # Build the smp model from the partial
        self.model = encoder_class()

        self.dice_loss = DiceLoss()
        self.bce_loss  = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _loss(self, logits: torch.Tensor, masks: torch.Tensor):
        masks = masks.float()
        l_dice = self.dice_loss(logits, masks)
        l_bce  = self.bce_loss(logits, masks)
        loss   = self.hparams.alpha * l_dice + (1.0 - self.hparams.alpha) * l_bce
        return loss, l_dice, l_bce

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss, l_dice, l_bce = self._loss(logits, masks)

        self.log("train/loss",      loss,   on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/dice_loss", l_dice, on_step=False, on_epoch=True)
        self.log("train/bce_loss",  l_bce,  on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss, l_dice, l_bce = self._loss(logits, masks)

        # Dice score (metric, not loss — higher is better)
        preds      = (torch.sigmoid(logits) > self.hparams.threshold).long()
        dice_score = dice(preds, masks.long(), ignore_index=0)

        self.log("val/loss",       loss,       on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/dice_score", dice_score, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/dice_loss",  l_dice,     on_step=False, on_epoch=True)
        self.log("val/bce_loss",   l_bce,      on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )
        return optimizer"""
PyTorch Lightning module for the baseline contrail segmentation model.

Instantiated via Hydra using _target_ + _partial_ pattern:

  encoder_class:
    _partial_: True
    _target_: segmentation_models_pytorch.Unet
    encoder_name: resnet50
    encoder_weights: ssl
    in_channels: 24
    classes: 1

The encoder_class partial is called inside __init__ to build the full model,
so all smp.Unet kwargs live in the YAML rather than this file.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.functional import dice


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs  = torch.sigmoid(logits)
        p_flat = probs.view(probs.size(0), -1)
        t_flat = targets.view(targets.size(0), -1).float()

        intersection = (p_flat * t_flat).sum(dim=1)
        score = (2.0 * intersection + self.smooth) / (
            p_flat.sum(dim=1) + t_flat.sum(dim=1) + self.smooth
        )
        return 1.0 - score.mean()


class PretrainedUNET(pl.LightningModule):
    def __init__(
        self,
        encoder_class: Callable[[], nn.Module],
        lr:        float = 1e-3,
        wd:        float = 1e-4,
        beta1:     float = 0.9,
        beta2:     float = 0.999,
        threshold: float = 0.5,
        alpha:     float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder_class"])

        # Build the smp model from the partial
        self.model = encoder_class()

        self.dice_loss = DiceLoss()
        self.bce_loss  = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
      
    def _loss(self, logits: torch.Tensor, masks: torch.Tensor):
        masks = masks.float()
        l_dice = self.dice_loss(logits, masks)
        l_bce  = self.bce_loss(logits, masks)
        loss   = self.hparams.alpha * l_dice + (1.0 - self.hparams.alpha) * l_bce
        return loss, l_dice, l_bce

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss, l_dice, l_bce = self._loss(logits, masks)

        self.log("train/loss",      loss,   on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/dice_loss", l_dice, on_step=False, on_epoch=True)
        self.log("train/bce_loss",  l_bce,  on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss, l_dice, l_bce = self._loss(logits, masks)

        # Dice score (metric, not loss — higher is better)
        preds      = (torch.sigmoid(logits) > self.hparams.threshold).long()
        dice_score = dice(preds, masks.long(), ignore_index=0)

        self.log("val/loss",       loss,       on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/dice_score", dice_score, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/dice_loss",  l_dice,     on_step=False, on_epoch=True)
        self.log("val/bce_loss",   l_bce,      on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )
        return optimizer
