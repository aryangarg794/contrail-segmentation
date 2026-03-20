import numpy as np
import pandas as pd 
import torch

from torch.utils.data import Dataset, DataLoader
from segmentation_models_pytorch.encoders import get_preprocessing_fn

from contrail_segmentation.data.utils import fake_color_img, get_mask, metadata, create_grid


class ContrailDataset(Dataset):
    """We use the preprocessing from the paper/from the kaggle notebook and then use 
    those for this dataset, input becomes (256, 256, 24) or (256, 256, 3) if only mask mode
    """
    
    def __init__(self, size=256, mask_only=False, transform=None):
        super().__init__()
        self.df_meta = metadata
        self.mask_only = mask_only
        self.transform = transform
        self.grid = torch.tensor(create_grid(size)).unsqueeze(0)
    
    def __len__(self):
        return len(self.df_meta)
    
    def __getitem__(self, index):
        record_id = self.df_meta.loc[index]['record_id']
        img = fake_color_img(record_id, get_mask_frame_only=self.mask_only).reshape(256, 256, -1).astype(np.float32)
        target = get_mask(record_id)
        
        if self.transform is not None:
            augmented = self.transform(image=img, target=target)
            img = augmented["image"]
            target = augmented["target"]
            
        img = torch.tensor(img).permute(2, 0, 1).float()
        target = torch.tensor(target).permute(2, 0, 1).float()
        target = torch.nn.functional.grid_sample(target.unsqueeze(0), self.grid, align_corners=False,
                                                 padding_mode='border', mode='bilinear').squeeze(0)
        
        return img, target
    
