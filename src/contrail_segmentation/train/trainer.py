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
import dill

from sklearn.model_selection import train_test_split
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass
from datetime import datetime
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger 
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader, Subset 

from contrail_segmentation.data.dataset import ContrailDataset
from contrail_segmentation.models.pretrained_unet import PretrainedUNET
from contrail_segmentation.models.unet import UNET
from contrail_segmentation.train.utils import find_best_threshold
from contrail_segmentation.data.plotting import plot_train_examples, plot_test_examples
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
    
    all_seed_metrics = []

    for seed in cfg.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        generator = torch.Generator().manual_seed(seed)
        
        train_transform = A.Compose([
            A.RandomRotate90(p=0.75),
            A.OneOf([
                A.HorizontalFlip(p=0.75),
                A.VerticalFlip(p=0.75),
            ], p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                rotate=(-90, 90),
                translate_percent={"x": (-0.0, 0.0), "y": (-0.0, 0.0)},
                shear=(-5, 5),
                p=0.5,
            ),
            A.GaussianBlur(sigma_limit=0.3, p=0.25),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.15, p=0.3),
        ], additional_targets={'pixel_mask': 'mask', 'soft_mask': 'mask'})
        
        timestamp = datetime.now().strftime("%d_%b_%Y__%Hh%Mm")
        name = cfg.run_name + f'_seed{seed}_{timestamp}'
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
            random_state=seed, 
            stratify=labels
        )

        val_idx, test_idx, _, _ = train_test_split(
            temp_idx, 
            temp_labels, 
            test_size=0.6, 
            random_state=seed, 
            stratify=temp_labels
        )
                
        train_set_base = ContrailDataset(
            mask_only=cfg.data.mask_only,
            y_fix=cfg.data.y_fix,
            transform=train_transform if cfg.data.transform else None,
            soft=cfg.data.soft
        )

        val_set_base = ContrailDataset(
            mask_only=cfg.data.mask_only, 
            y_fix=cfg.data.y_fix, 
            val=True,
            soft=cfg.data.soft
        )

        test_set_base = ContrailDataset(
            mask_only=cfg.data.mask_only, 
            y_fix=cfg.data.y_fix,
            val=True,
            soft=cfg.data.soft
        )

        train_loader = DataLoader(
            Subset(train_set_base, train_idx),
            batch_size=cfg.data.batch_size,
            generator=generator,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        val_loader = DataLoader(
            Subset(val_set_base, val_idx), 
            batch_size=64, 
            generator=generator,
            shuffle=False, 
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

        test_loader = DataLoader(
            Subset(test_set_base, test_idx), 
            batch_size=64, 
            generator=generator,
            shuffle=False, 
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        
        callbacks = [
            LearningRateMonitor('step'),
            EarlyStopping(monitor="val/dice", patience=30, mode="max"),
            ModelCheckpoint(monitor="val/dice", mode="max", save_top_k=1, 
                            filename=f"{cfg.run_name}_best-contrail_seed{seed}",
                            dirpath="checkpoints")
        ]
        model = instantiate(cfg.model)(cfg.data.soft)
        trainer = Trainer(**cfg.trainer, logger=logger, callbacks=callbacks)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        best_model_path = trainer.checkpoint_callback.best_model_path
        if best_model_path:
            print(f"Loading best weights into current model: {best_model_path}")
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['state_dict'])
            
            model.to('cuda')
            model.eval() 
        else:
            print("Warning: No best checkpoint found, using last weights.")
            model.to('cuda')
        
        best_thresh_pos = find_best_threshold(model, val_loader, device='cuda', soft=cfg.data.soft, pos_only=True)
        best_thresh_dice = find_best_threshold(model, val_loader, device='cuda', soft=cfg.data.soft, pos_only=False)
        model.threshold = best_thresh_dice
        model.mask_only = cfg.data.mask_only

        plot_train_examples(model, logger, train_loader, mask_only=cfg.data.mask_only, soft=cfg.data.soft)
        test_metrics = trainer.test(model, dataloaders=test_loader, ckpt_path="best")
        plot_test_examples(model, logger, test_loader, threshold=best_thresh_pos, mask_only=cfg.data.mask_only, soft=cfg.data.soft)
        all_seed_metrics.append(test_metrics[0])
        
        torch.cuda.empty_cache()
        wandb.finish()
    
    results = {}
    for key in all_seed_metrics[0].keys():
        values = [m[key] for m in all_seed_metrics]
        results[f"{key}_mean"] = np.mean(values)
        results[f"{key}_std"] = np.std(values)
    
    print("\nFinal Results across seeds:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    results = {}
    for key in all_seed_metrics[0].keys():
        values = [m[key] for m in all_seed_metrics]
        results[f"{key}_mean"] = float(np.mean(values))
        results[f"{key}_std"] = float(np.std(values))
    
    os.makedirs("results", exist_ok=True)
    base_name = f"{cfg.run_name}_{timestamp}"
    
    dill_path = os.path.join("results", f"{base_name}.dill")
    with open(dill_path, 'wb') as f:
        dill.dump(results, f)
        
    return results

if __name__ == "__main__":
    main()