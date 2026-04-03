"""
train.py — Hydra entry-point for contrail segmentation.

Usage
-----
uv run train model=baseline_unet
uv run train model=baseline_unet model.lr=1e-4
uv run train model=baseline_unet model.encoder_class.encoder_name=resnet34

Config files live in conf/
  conf/config.yaml          ← top-level defaults
  conf/model/baseline_unet.yaml
"""

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader

from contrail_segmentation.data.dataset import build_datasets


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:

    pl.seed_everything(cfg.seed, workers=True)

    # ── Model (instantiated from the chosen model yaml) ───────────────
    model = instantiate(cfg.model)

    # ── Data ──────────────────────────────────────────────────────────
    train_ds, val_ds = build_datasets(
        root_dir=cfg.data.root_dir,
        metadata_csv=cfg.data.metadata_csv,
        time_step=cfg.data.time_step,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    # ── Callbacks ─────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            monitor="val/dice_score",
            mode="max",
            save_top_k=1,
            filename="best-{epoch:03d}-{val/dice_score:.4f}",
        ),
        EarlyStopping(
            monitor="val/dice_score",
            mode="max",
            patience=cfg.trainer.patience,
            verbose=True,
        ),
    ]

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=callbacks,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    train()