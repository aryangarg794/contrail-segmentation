import numpy as np
import pandas as pd 
import torch
import h5py

from torch.utils.data import Dataset, DataLoader

from contrail_segmentation.data.utils import (
    fake_color_img, 
    get_mask, 
    get_mask_ind, 
    metadata, 
    shift, 
    DATA_DIR,
    get_ash_image
)


class ContrailDataset(Dataset):
    """We use the preprocessing from the paper/from the kaggle notebook and then use 
    those for this dataset, input becomes (256, 256, 24) or (256, 256, 3) if only mask mode
    """
    
    def __init__(self, mask_only=False, y_fix=False, soft=False, transform=None):
        super().__init__()
        self.df_meta = metadata
        self.mask_only = mask_only
        self.soft = soft
        self.transform = transform
        self.y_fix = y_fix
        self.file = None
        
    
    def __len__(self):
        return len(self.df_meta)
    
    def __getitem__(self, index):
        record_id = self.df_meta.loc[index]['record_id']
        img = get_ash_image(record_id, get_mask_only=self.mask_only).reshape(256, 256, -1).astype(np.float32)
        
        if self.soft:
            target = np.mean(get_mask_ind(record_id), axis=3, keepdims=True)
        else:
            target = get_mask(record_id)

        if self.y_fix:
            img = shift(img)

        if self.transform is not None:
            augmented = self.transform(image=img, target=target)
            img = augmented["image"]
            target = augmented["target"]

        img = torch.tensor(img).permute(2, 0, 1).float()
        target = torch.tensor(target).permute(2, 0, 1).float()
        
        return img, target
    
    # def __getitem__(self, index):
    #     if self.file is None: # Only open when the worker actually starts
    #         self.file = h5py.File(DATA_DIR + '/train_dataset.h5', 'r')

    #     record_id = str(self.df_meta.loc[index]['record_id'])
    #     img = self.file[record_id]['image'][()].reshape(256, 256, -1).astype(np.float32)
    #     if self.soft:
    #         target = np.mean(self.file[record_id]['individual_masks'][()], axis=3, keepdims=True)
    #     else:
    #         target = self.file[record_id]['pixel_mask'][()]

    #     if self.y_fix:
    #         img = shift(img)

    #     if self.transform is not None:
    #         augmented = self.transform(image=img, target=target)
    #         img = augmented["image"]
    #         target = augmented["target"]

    #     img = torch.tensor(img).permute(2, 0, 1).float()
    #     target = torch.tensor(target).permute(2, 0, 1).float()
        
    #     return img, target
