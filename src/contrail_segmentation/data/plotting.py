import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import io 
import wandb

from PIL import Image
from contrail_segmentation.data.utils import get_mask, fake_color_img, metadata, get_ash_image

def plot_train_examples(model: nn.Module, logger, train_loader, device: str = 'cuda', num_values: int = 5,
                        mask_only: bool = False):
    inputs = []
    outputs = []
    found_empty = False
    
    for batch in train_loader:
        imgs, targets = batch
        for i in range(imgs.size(0)):
            has_contrail = targets[i].sum() > 0
            
            if not has_contrail and not found_empty:
                inputs.append(imgs[i])
                outputs.append(targets[i])
                found_empty = True
            elif has_contrail and len(inputs) < num_values:
            
                if not found_empty or (found_empty and len(inputs) < num_values):
                    inputs.append(imgs[i])
                    outputs.append(targets[i])
            
            if len(inputs) == num_values:
                break
        if len(inputs) == num_values:
            break

    fig, axes = plt.subplots(len(inputs), 4, figsize=(20, 10))
    if len(inputs) == 1: axes = axes.reshape(1, -1)
    sigmoid = nn.Sigmoid()
    
    for i, inp in enumerate(inputs):
        true_mask = outputs[i]
        y_hat = model.model(inp.unsqueeze(0))
        inp = inp.permute(1, 2, 0).view(256, 256, 3, 8) if not mask_only else inp.permute(1, 2, 0)
        true_mask = true_mask.squeeze()
        axes[i, 0].imshow(true_mask)
        axes[i, 1].imshow(inp[:, :, :, 4] if not mask_only else inp[:, :, :])
        
        y_hat = sigmoid(y_hat).view(256, 256).float().detach().cpu()
        axes[i, 2].imshow(y_hat.numpy(), vmin=0, vmax=1)
        axes[i, 3].imshow((y_hat > model.threshold).numpy())
        for j in range(4):
            axes[i, j].axis('off')
    
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    
    logger.experiment.log({'Train Examples': wandb.Image(img)})
    plt.close(fig)
    

def plot_examples(model: nn.Module, idxs: list, device: str = 'cuda', mask_only = False):
    inputs = [get_ash_image(metadata.loc[idx]['record_id'], get_mask_only=mask_only) for idx in idxs]
    fig, axes = plt.subplots(5, 4, figsize=(20, 10))
    sigmoid = nn.Sigmoid()
    
    for i, inp in enumerate(inputs):
        idx = idxs[i]
        true_mask = get_mask(metadata.loc[idx]['record_id'])
        axes[i, 0].imshow(true_mask)
        axes[i, 1].imshow(inp[:, :, :, 4] if not mask_only else inp[:, :, :])
        
        torch_inp = torch.tensor(inp).view(1, -1, 256, 256).to(device)
        y_hat = model.model(torch_inp)
        y_hat = sigmoid(y_hat).view(256, 256).float().cpu()
        axes[i, 2].imshow(y_hat.numpy(), vmin=0, vmax=1)
        axes[i, 3].imshow((y_hat > model.threshold).numpy())
        for j in range(4):
            axes[i, j].axis('off')
    
    fig.tight_layout()
    
    return fig, axes