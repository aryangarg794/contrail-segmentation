import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from contrail_segmentation.data.utils import TEST_IDXS, get_mask, fake_color_img, metadata

def plot_val_examples(model: nn.Module, device: str = 'cuda'):
    inputs = [fake_color_img(metadata.loc[idx]['record_id'], get_mask_frame_only=False) for idx in TEST_IDXS]
    fig, axes = plt.subplots(5, 4, figsize=(20, 10))
    sigmoid = nn.Sigmoid()
    
    for i, inp in enumerate(inputs):
        idx = TEST_IDXS[i]
        true_mask = get_mask(metadata.loc[idx]['record_id'])
        axes[i, 0].imshow(true_mask)
        axes[i, 1].imshow(inp[:, :, :, 4])
        
        torch_inp = torch.tensor(inp).view(1, -1, 256, 256).to(device)
        y_hat = model.model(torch_inp)
        y_hat = sigmoid(y_hat).view(256, 256)
        axes[i, 2].imshow((y_hat > 0.5).float().cpu().numpy())
        axes[i, 3].imshow((y_hat > model.threshold).float().cpu().numpy())
        for j in range(4):
            axes[i, j].axis('off')
    
    fig.tight_layout()
    
    return fig, axes