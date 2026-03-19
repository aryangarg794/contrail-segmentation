import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from contrail_segmentation.data.utils import get_mask, fake_color_img, metadata

def plot_examples(model: nn.Module, idxs: list, device: str = 'cuda', mask_only = False):
    inputs = [fake_color_img(metadata.loc[idx]['record_id'], get_mask_frame_only=mask_only) for idx in idxs]
    fig, axes = plt.subplots(5, 4, figsize=(20, 10))
    sigmoid = nn.Sigmoid()
    
    for i, inp in enumerate(inputs):
        idx = idxs[i]
        true_mask = get_mask(metadata.loc[idx]['record_id'])
        axes[i, 0].imshow(true_mask)
        axes[i, 1].imshow(inp[:, :, :, 4] if not mask_only else inp[:, :, :])
        
        torch_inp = torch.tensor(inp).view(1, -1, 256, 256).to(device)
        y_hat = model.model(torch_inp)
        y_hat = sigmoid(y_hat).view(256, 256)
        axes[i, 2].imshow((y_hat > 0.5).float().cpu().numpy())
        axes[i, 3].imshow((y_hat > model.threshold).float().cpu().numpy())
        for j in range(4):
            axes[i, j].axis('off')
    
    fig.tight_layout()
    
    return fig, axes