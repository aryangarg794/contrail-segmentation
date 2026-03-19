import argparse
import albumentations as A
import numpy as np 
import os
import hydra
import lightning as pl
import random
import segmentation_models_pytorch as smp
import torch
import yaml

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from datetime import datetime
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger 
from torch.utils.data import DataLoader, Subset 

from contrail_segmentation.data.dataset import ContrailDataset
from contrail_segmentation.models.pretrained_unet import PretrainedUNET
from contrail_segmentation.models.unet import UNET
from contrail_segmentation.train.utils import find_best_threshold

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
        A.HorizontalFlip(p=0.4),
        A.RandomResizedCrop(size=(256, 256), scale=(0.75, 1.0), p=0.4)
    ])
    
    timestamp = datetime.now().strftime("%d_%b_%Y__%Hh%Mm")
    name = cfg.run_name + f'_seed{cfg.seed}_{timestamp}'
    group_name = cfg.run_name + "_" + timestamp    
    wandb_dict = OmegaConf.to_container(cfg.wandb, resolve=True, throw_on_missing=True)
    logger = WandbLogger(**wandb_dict, name=name, group=group_name)
    
    print(OmegaConf.to_yaml(cfg, resolve=True))
            
    dataset = ContrailDataset(mask_only=cfg.data.mask_only)
    
    indices = np.arange(len(dataset))
    train_size = int(0.8 * len(dataset))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_set = ContrailDataset(
        mask_only=cfg.data.mask_only, 
        transform=train_transform if cfg.data.transform else None  
    )
    test_set = ContrailDataset(
        mask_only=cfg.data.mask_only    
    )
    
    train_loader = DataLoader(Subset(train_set, train_indices), batch_size=cfg.data.batch_size, generator=generator, 
                              shuffle=True, pin_memory=True)
    test_loader = DataLoader(Subset(test_set, test_indices), batch_size=cfg.data.batch_size, pin_memory=True)
    
    model = instantiate(cfg.model)
    trainer = Trainer(**cfg.trainer, logger=logger)
    trainer.fit(model, train_dataloaders=train_loader)
    
    best_thresh = find_best_threshold(model, test_loader)
    model.threshold = best_thresh
    model.mask_only = cfg.data.mask_only
    
    test_metrics = trainer.test(model, dataloaders=test_loader)
    
    torch.cuda.empty_cache()
    
    return test_metrics


if __name__ == "__main__":
    main()
    

     