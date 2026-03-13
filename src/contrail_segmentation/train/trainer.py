import argparse
import albumentations as A
import numpy as np 
import os
import lightning as pl
import random
import segmentation_models_pytorch as smp
import torch
import yaml

from dataclasses import dataclass
from datetime import datetime
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger 
from torch.utils.data import DataLoader, Subset


from contrail_segmentation.data.dataset import ContrailDataset
from contrail_segmentation.models.pretrained_unet import PretrainedUNET
from contrail_segmentation.models.unet import UNET
from contrail_segmentation.train.utils import find_best_threshold

@dataclass
class Config:
    run_name: str 
    batch_size: int
    only_mask: bool
    model_type: str
    epochs: int 
    device: str 
    seed: int
    transform: bool 


def main(cfg: Config):
    
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
    with open('src/contrail_segmentation/config/wandb.yaml', 'r') as file:
        wandb_dict = yaml.safe_load(file)
        file.close()
    group_name = cfg.run_name + "_" + timestamp    
    logger = WandbLogger(**wandb_dict, name=name, group=group_name)
    
    match cfg.model_type:
        case 'pretrained_unet':
            model = PretrainedUNET()  
        case 'unet':
            model = UNET()
            
    dataset = ContrailDataset(mask_only=cfg.only_mask)
    
    indices = np.arange(len(dataset))
    train_size = int(0.8 * len(dataset))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_set = ContrailDataset(
        mask_only=cfg.only_mask, 
        transform=train_transform if cfg.transform else None  
    )
    test_set = ContrailDataset(
        mask_only=cfg.only_mask    
    )
    
    train_loader = DataLoader(Subset(train_set, train_indices), batch_size=cfg.batch_size, generator=generator, 
                              shuffle=True, pin_memory=True)
    test_loader = DataLoader(Subset(test_set, test_indices), batch_size=cfg.batch_size, pin_memory=True)
    
    accelerator = 'gpu' if cfg.device == 'cuda' else 'cpu'
    trainer = Trainer(accelerator=accelerator, max_epochs=cfg.epochs, logger=logger, log_every_n_steps=1)
    trainer.fit(model, train_dataloaders=train_loader)
    
    best_thresh = find_best_threshold(model, test_loader)
    model.threshold = best_thresh
    
    test_metrics = trainer.test(model, dataloaders=test_loader)
    
    torch.cuda.empty_cache()
    
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', type=int, default=32, help='batch size')
    parser.add_argument('-m', '--model', type=str, default='pretrained_unet', help='model type')
    parser.add_argument('-r', '--name', type=str, default='pretrained_unet_fake_rgb', help='run name')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='device')
    parser.add_argument('-o', '--mask', action='store_true', help='render mode')
    parser.add_argument('-t', '--trans', action='store_false', help='transform mode')
    parser.add_argument('-s', '--seed', type=int, default=0, help='seed')
    
    with open('src/contrail_segmentation/config/optim/adam.yaml', 'r') as file:
        opt_config = yaml.safe_load(file)
        file.close()
    
    args = parser.parse_args()
    config = Config(
        run_name=args.name, 
        batch_size=args.batch, 
        epochs=opt_config['epochs'], 
        model_type=args.model,
        device=args.device, 
        only_mask=args.mask, 
        seed=args.seed,
        transform=args.trans
    )
    
    main(config)
    

     