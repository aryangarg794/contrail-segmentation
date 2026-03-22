import os
import numpy as np
import pandas as pd

DATA_DIR = '/media/nicomoft/Stuff/CV/final_project/train'
META_PATH = '/media/nicomoft/Stuff/CV/final_project/train_metadata.json'

def get_band_images(idx: str, parent_folder: str, band: str):
    idx = str(idx)
    return np.load(os.path.join(parent_folder, idx, f'band_{band}.npy'))

def get_mask(idx: str, parent_folder: str = DATA_DIR):
    idx = str(idx)
    return np.load(os.path.join(parent_folder, idx, f'human_pixel_masks.npy'))

_T11_BOUNDS = (243, 303)
_CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
_TDIFF_BOUNDS = (-4, 2)

metadata = pd.read_json(META_PATH)

def normalize_range(data, bounds):
    return (data - bounds[0]) / (bounds[1] - bounds[0])

def fake_color_img(idx, parent_folder = DATA_DIR, get_mask_frame_only=False):
    
    band11 = get_band_images(idx, parent_folder, '11')
    band14 = get_band_images(idx, parent_folder, '14')
    band15 = get_band_images(idx, parent_folder, '15')
    
    if get_mask_frame_only:
        band11 = band11[:,:,4]
        band14 = band14[:,:,4]
        band15 = band15[:,:,4]

    r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
    g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize_range(band14, _T11_BOUNDS)
    false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)
    return false_color

TEST_IDXS = [1228, 947, 1376, 1340, 826]