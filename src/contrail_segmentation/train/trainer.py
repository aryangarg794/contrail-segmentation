import argparse
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
from torch.utils.data import DataLoader, random_split


from contrail_segmentation.data.dataset import ContrailDataset
from contrail_segmentation.models.pretrained_unet import PretrainedUNET

@dataclass
class Config:
    run_name: str 
    batch_size: int
    only_mask: bool
    model_type: str
    epochs: int 
    device: str 
    seed: int


def main(cfg: Config):
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    
    generator = torch.Generator().manual_seed(cfg.seed)
    
    timestamp = datetime.now().strftime("%d_%b_%Y__%Hh%Mm")
    name = cfg.run_name + f'_seed_{timestamp}'
    print(os.getcwd())
    with open('src/contrail_segmentation/config/wandb.yaml', 'r') as file:
        wandb_dict = yaml.safe_load(file)
        file.close()
    group_name = cfg.run_name + "_" + timestamp    
    logger = WandbLogger(**wandb_dict, name=name, group=group_name)
    
    match cfg.model_type:
        case 'pretrained_unet':
            model = PretrainedUNET()  
            
    dataset = ContrailDataset(mask_only=cfg.only_mask)
    train_set, test_set = random_split(dataset, [0.8, 0.2], generator=generator)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, generator=generator, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, pin_memory=True)
    
    accelerator = 'gpu' if cfg.device == 'cuda' else 'cpu'
    trainer = Trainer(accelerator=accelerator, max_epochs=cfg.epochs, logger=logger, log_every_n_steps=1)
    trainer.fit(model, train_dataloaders=train_loader)
    
    test_metrics = trainer.test(model, dataloaders=test_loader)
    
    return test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--epochs', type=int, default=25, help='timesteps')
    parser.add_argument('-b', '--batch', type=int, default=32, help='batch size')
    parser.add_argument('-m', '--model', type=str, default='pretrained_unet', help='model type')
    parser.add_argument('-r', '--name', type=str, default='pretrained_unet_fake_rgb', help='run name')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='device')
    parser.add_argument('-o', '--mask', action='store_true', help='render mode')
    parser.add_argument('-s', '--seed', type=int, default=0, help='seed')
    
    args = parser.parse_args()
    config = Config(
        run_name=args.name, 
        batch_size=args.batch, 
        epochs=args.epochs, 
        model_type=args.model,
        device=args.device, 
        only_mask=args.mask, 
        seed=args.seed
    )
    
    main(config)
    

     