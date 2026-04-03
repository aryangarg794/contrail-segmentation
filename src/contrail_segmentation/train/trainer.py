import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from contrail_segmentation.data.dataset import ContrailDataset

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:

    pl.seed_everything(cfg.seed, workers=True)

    model = instantiate(cfg.model)

    train_ds = ContrailDataset(val=False)
    val_ds   = ContrailDataset(val=True)

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

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=callbacks,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()