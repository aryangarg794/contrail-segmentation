import argparse
import albumentations as A
import numpy as np 
import os
import wandb
import hydra
import lightning as pl
import random
import segmentation_models_pytorch as smp
import torch
import yaml
from sklearn.model_selection import train_test_split
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from datetime import datetime
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger 
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from torch.utils.data import DataLoader, Subset 

from contrail_segmentation.data.dataset import ContrailDataset
from contrail_segmentation.models.pretrained_unet import PretrainedUNET
from contrail_segmentation.models.unet import UNET
from contrail_segmentation.train.utils import find_best_threshold
from contrail_segmentation.data.utils import metadata

import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings(
    "ignore", ".*can be accelerated via the 'torch-scatter' package*"
)

@hydra.main(version_base=None, config_path='../config', config_name='default')
def main(cfg: DictConfig):
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    
    generator = torch.Generator().manual_seed(cfg.seed)
    
    train_transform = A.Compose([
        A.ShiftScaleRotate(
            scale_limit=0.2,
            rotate_limit=0,
            shift_limit=0.3,
            border_mode=0,
            value=0,
            p=0.5,
        ),
        A.GaussNoise(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.15, p=0.5),
    ])
    
    timestamp = datetime.now().strftime("%d_%b_%Y__%Hh%Mm")
    name = cfg.run_name + f'_seed{cfg.seed}_{timestamp}'
    group_name = cfg.run_name + "_" + timestamp    
    wandb_dict = OmegaConf.to_container(cfg.wandb, resolve=True, throw_on_missing=True)
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger = WandbLogger(**wandb_dict, name=name, group=group_name, config=config_dict)

    indices = np.arange(len(metadata))
    labels = metadata['label'].values

    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, 
        labels, 
        test_size=0.25, 
        random_state=cfg.seed, 
        stratify=labels
    )

    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, 
        temp_labels, 
        test_size=0.6, 
        random_state=cfg.seed, 
        stratify=temp_labels
    )
            
    train_set_base = ContrailDataset(
        mask_only=cfg.data.mask_only,
        y_fix=cfg.data.y_fix,
        transform=train_transform if cfg.data.transform else None
    )

    val_set_base = ContrailDataset(
        mask_only=cfg.data.mask_only, 
        y_fix=cfg.data.y_fix
    )

    test_set_base = ContrailDataset(
        mask_only=cfg.data.mask_only, 
        y_fix=cfg.data.y_fix
    )

    train_loader = DataLoader(
        Subset(train_set_base, train_idx),
        batch_size=cfg.data.batch_size,
        generator=generator,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        Subset(val_set_base, val_idx), 
        batch_size=64, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        Subset(test_set_base, test_idx), 
        batch_size=64, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    model = instantiate(cfg.model)
    trainer = Trainer(**cfg.trainer, logger=logger, callbacks=[LearningRateMonitor('step')])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    best_thresh = find_best_threshold(model, val_loader, device='cuda')
    model.threshold = best_thresh
    model.mask_only = cfg.data.mask_only
    
    test_metrics = trainer.test(model, dataloaders=test_loader)
    
    torch.cuda.empty_cache()
    wandb.finish()
    
    return test_metrics

if __name__ == "__main__":
    main()