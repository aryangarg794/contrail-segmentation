import numpy as np
import torch

from torch.utils.data import Dataset

from contrail_segmentation.data.utils import fake_color_img, get_mask, metadata


class ContrailDataset(Dataset):
    """We use the preprocessing from the paper/from the kaggle notebook and then use
    those for this dataset, input becomes (256, 256, 24) or (256, 256, 3) if only mask mode
    """

    def __init__(self, mask_only=False, transform=None):
        super().__init__()
        self.df_meta = metadata
        self.mask_only = mask_only
        self.transform = transform

    def __len__(self):
        return len(self.df_meta)

    def __getitem__(self, index):
        record_id = self.df_meta.loc[index]['record_id']
        img = fake_color_img(record_id, get_mask_frame_only=self.mask_only).reshape(256, 256, -1).astype(np.float32)
        target = get_mask(record_id).astype(np.float32)

        if target.ndim == 2:
            target = target[..., None]

        if self.transform is not None:
            augmented = self.transform(image=img, mask=target)
            img = augmented["image"]
            target = augmented["mask"]

        if target.ndim == 2:
            target = target[..., None]

        img = torch.tensor(img).permute(2, 0, 1).float()
        target = torch.tensor(target).permute(2, 0, 1).float()
        return img, target
    
