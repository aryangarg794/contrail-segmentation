import numpy as np
import pandas as pd 
import torch

from torch.utils.data import Dataset, DataLoader

from contrail_segmentation.data.utils import DATA_DIR, META_PATH, fake_color_img, get_mask


class ContrailDataset(Dataset):
    """We use the preprocessing from the paper/from the kaggle notebook and then use 
    those for this dataset, input becomes (256, 256, 24) or (256, 256, 3) if only mask mode
    """
    
    def __init__(self, mask_only=False):
        super().__init__()
        self.df_meta = pd.read_json(META_PATH)
        self.mask_only = mask_only
    
    def __len__(self):
        return len(self.df_meta)
    
    def __getitem__(self, index):
        record_id = self.df_meta.loc[index]['record_id']
        img = torch.tensor(fake_color_img(record_id, DATA_DIR, get_mask_frame_only=self.mask_only)).view(-1, 256, 256).float()
        target = torch.tensor(get_mask(record_id, DATA_DIR)).permute(2, 0, 1).float()
        return img, target
    
