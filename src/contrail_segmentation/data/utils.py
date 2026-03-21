import os
import numpy as np
import pandas as pd
import cv2
import h5py

DATA_DIR = 'data'
TRAIN_DIR = 'data/train'
META_PATH = 'data/train_metadata.csv'

# from https://github.com/divideconcept/fastnumpyio
def load(file):
    file=open(file,"rb")
    header = file.read(128)
    descr = str(header[19:25], 'utf-8').replace("'","").replace(" ","")
    shape = tuple(int(num) for num in str(header[60:120], 'utf-8').replace(', }', '').replace('(', '').replace(')', '').split(','))
    datasize = np.lib.format.descr_to_dtype(descr).itemsize
    for dimension in shape:
        datasize *= dimension
    return np.ndarray(shape, dtype=descr, buffer=file.read(datasize))

def get_mask_ind(idx: str, parent_folder: str = TRAIN_DIR):
    idx = str(idx)
    return load(os.path.join(parent_folder, idx, f'human_individual_masks.npy'))

def get_band_images(idx: str, parent_folder: str, band: str):
    idx = str(idx)
    return load(os.path.join(parent_folder, idx, f'band_{band}.npy'))

def get_mask(idx: str, parent_folder: str = TRAIN_DIR):
    idx = str(idx)
    return load(os.path.join(parent_folder, idx, f'human_pixel_masks.npy'))

def get_ash_image(idx: str, parent_folder: str = TRAIN_DIR, get_mask_only: bool = False):
    idx = str(idx)
    img = load(os.path.join(parent_folder, idx, f'ash_color_img.npy'))
    return img if not get_mask_only else img[:, :, :, 4]

_T11_BOUNDS = (243, 303)
_CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
_TDIFF_BOUNDS = (-4, 2)

metadata = pd.read_csv(META_PATH)
# train_file = h5py.File(DATA_DIR+'/train_dataset.h5', 'r')

def normalize_range(data, bounds):
    return (data - bounds[0]) / (bounds[1] - bounds[0])

def fake_color_img(idx, parent_folder = TRAIN_DIR, get_mask_frame_only=False):
    
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


def shift(img):
    shift_matrix = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]], dtype=np.float32)

    # Apply the affine transformation using cv2.warpAffine
    shifted_img = cv2.warpAffine(
        img,
        shift_matrix,
        (img.shape[1], img.shape[0]), # Keep the same size
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return shifted_img.astype(img.dtype)

TEST_IDXS = [1228, 947, 1376, 1340, 826]