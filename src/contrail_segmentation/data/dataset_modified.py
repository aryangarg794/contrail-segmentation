# import numpy as np
# import pandas as pd 
# import torch

# from torch.utils.data import Dataset, DataLoader

# from contrail_segmentation.data.utils import fake_color_img, get_mask, metadata


# class ContrailDataset(Dataset):
#     """We use the preprocessing from the paper/from the kaggle notebook and then use 
#     those for this dataset, input becomes (256, 256, 24) or (256, 256, 3) if only mask mode
#     """
    
#     def __init__(self, mask_only=False):
#         super().__init__()
#         self.df_meta = metadata
#         self.mask_only = mask_only
    
#     def __len__(self):
#         return len(self.df_meta)
    
#     def __getitem__(self, index):
#         record_id = self.df_meta.loc[index]['record_id']
#         img = torch.tensor(fake_color_img(record_id, get_mask_frame_only=self.mask_only)).view(256, 256, -1).permute(2, 0, 1).float()
#         target = torch.tensor(get_mask(record_id)).permute(2, 0, 1).float()
#         return img, target
    



import numpy as np
import torch

from torch.utils.data import Dataset

from contrail_segmentation.data.utils import fake_color_img, get_mask, metadata


def random_flip_rotate(img, mask):
    """
    img:  H x W x C
    mask: H x W x C  (or H x W x 1)
    """

    # horizontal flip
    if np.random.rand() < 0.5:
        img = np.flip(img, axis=1).copy()
        mask = np.flip(mask, axis=1).copy()

    # vertical flip
    if np.random.rand() < 0.5:
        img = np.flip(img, axis=0).copy()
        mask = np.flip(mask, axis=0).copy()

    # rotate by 0, 90, 180, 270 degrees
    k = np.random.randint(0, 4)
    img = np.rot90(img, k, axes=(0, 1)).copy()
    mask = np.rot90(mask, k, axes=(0, 1)).copy()

    return img, mask


def random_exposure(img, low=0.9, high=1.1):
    """
    Mild brightness/intensity scaling.
    Works better than aggressive exposure for satellite-like inputs.
    """
    scale = np.random.uniform(low, high)
    img = img * scale
    return img


def random_noise(img, std=0.01):
    """
    Add mild Gaussian noise to the image only.
    """
    noise = np.random.normal(loc=0.0, scale=std, size=img.shape)
    img = img + noise
    return img


def augment_sample(img, mask, p_exposure=0.5, p_noise=0.3):
    """
    Apply geometric transforms to both image and mask,
    and intensity transforms only to image.
    """
    # img, mask = random_flip_rotate(img, mask)

    if np.random.rand() < p_exposure:
        img = random_exposure(img, low=0.9, high=1.1)

    if np.random.rand() < p_noise:
        img = random_noise(img, std=0.01)

    # keep values sensible
    img = np.clip(img, a_min=0.0, a_max=None)

    return img, mask


class ContrailDataset(Dataset):
    """
    We use the preprocessing from the paper / Kaggle notebook.
    Input becomes (256, 256, 24) or (256, 256, 3) if only mask mode.
    """

    def __init__(self, mask_only=False, augment=False):
        super().__init__()
        self.df_meta = metadata
        self.mask_only = mask_only
        self.augment = augment

    def __len__(self):
        return len(self.df_meta)

    def __getitem__(self, index):
        record_id = self.df_meta.loc[index]["record_id"]

        # Load as NumPy first
        img = fake_color_img(record_id, get_mask_frame_only=self.mask_only)
        target = get_mask(record_id)

        # Ensure expected shapes
        img = np.asarray(img).reshape(256, 256, -1)
        target = np.asarray(target)

        # If target comes as H x W, make it H x W x 1
        if target.ndim == 2:
            target = target[..., None]

        # Augment before converting to torch
        if self.augment:
            img, target = augment_sample(img, target)

        # Convert to torch tensors
        img = torch.tensor(img).permute(2, 0, 1).float()
        target = torch.tensor(target).permute(2, 0, 1).float()

        return img, target
