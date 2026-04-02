import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import io 
import wandb

from PIL import Image
from contrail_segmentation.data.utils import get_mask, fake_color_img, metadata, get_ash_image

def plot_train_examples(model: nn.Module, logger, train_loader, device: str = 'cuda', num_values: int = 5,
                        mask_only: bool = False, soft: bool = False):
    inputs = []
    outputs = []
    found_empty = False
    
    for batch in train_loader:
        if soft:
            imgs, targets, _ = batch
        else:
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
        y_hat = model.model(inp.unsqueeze(0).to(device))
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

def find_diverse_samples(model, dataloader, threshold, k=3, soft=False, device='cuda'):
    diverse_idxs = {} 
    best_samples = [] 
    worst_dice = 1.1
    model.eval()
    model.to(device)
    
    for batch in dataloader:
        if soft:
            x, y, _ = batch
        else:
            x, y = batch 
        
        with torch.no_grad():
            logits = model.model(x.to(device))
            probs = torch.sigmoid(logits).squeeze(1) 
            preds = (probs > threshold).float()
            
        y_true = y.to(device).squeeze(1) 
        
        for i in range(x.shape[0]):
            img, mask = x[i], y[i]
            pixel_count = mask.sum().item()
            
            inter = (preds[i] * y_true[i]).sum()
            total = preds[i].sum() + y_true[i].sum()
            sample_dice = (2.0 * inter / (total + 1e-7)).item()

            if pixel_count > 100:
                best_samples.append((sample_dice, img, mask))
                best_samples = sorted(best_samples, key=lambda x: x[0], reverse=True)[:k]
            
            if sample_dice < worst_dice and pixel_count > 100:
                worst_dice = sample_dice
                diverse_idxs['worst'] = (img, mask, f"Worst Match (Dice: {sample_dice:.2f})")

            if 'large' not in diverse_idxs and pixel_count > 800:
                diverse_idxs['large'] = (img, mask, "Large/Clear")
        
            if 'ghost' not in diverse_idxs and 0 < pixel_count < 50:
                diverse_idxs['ghost'] = (img, mask, "Ultra-Thin/Ghost")
        
            if 'mid' not in diverse_idxs and 400 <= pixel_count <= 800:
                diverse_idxs['mid'] = (img, mask, "Standard Case")
                
            if 'empty' not in diverse_idxs and pixel_count == 0:
                diverse_idxs['empty'] = (img, mask, "Clear Sky (Neg)")

            if 'noisy_empty' not in diverse_idxs and pixel_count == 0:
                if img.std() > 0.5: 
                    diverse_idxs['noisy_empty'] = (img, mask, "Noisy Sky (Neg)")

        if len(diverse_idxs) >= 5 and len(best_samples) >= 3: break
        
    final_samples = []
    for rank, (d, img, m) in enumerate(best_samples):
        final_samples.append((img, m, f"Best #{rank+1} (Dice: {d:.2f})"))
    
    final_samples.extend(list(diverse_idxs.values()))
    return final_samples

def plot_examples(model, dataloader, threshold, soft=False, mask_only=True, device='cuda'):
    samples = find_diverse_samples(model, dataloader, threshold, soft=soft, device=device)
    num_samples = len(samples)
    
    fig, axes = plt.subplots(4, num_samples, figsize=(4 * num_samples, 14))
    model.to(device).eval()
    
    row_titles = ["Normalized Ash", "Ground Truth", "Confidence (Heatmap)", f"Pred (T={threshold:.2f})"]
    
    for i, (img, mask, label) in enumerate(samples):
        with torch.no_grad():
            logits = model.model(img.unsqueeze(0).to(device))
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        img_np = img.permute(1, 2, 0).cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()
        img_plot = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-7)
    
        axes[0, i].imshow(img_plot)
        axes[0, i].set_title(label, fontsize=12, fontweight='bold')
        
        axes[1, i].imshow(mask_np, cmap='gray')
        axes[2, i].imshow(probs, cmap='magma', vmin=0, vmax=1)
        axes[3, i].imshow(probs > threshold, cmap='gray')

    for r, title in enumerate(row_titles):
        axes[r, 0].set_ylabel(title, fontsize=14, fontweight='bold', labelpad=20)

    for ax in axes.flatten():
        ax.set_xticks([]); ax.set_yticks([])
        
    plt.tight_layout()
    return fig, axes

def plot_test_examples(model, logger, dataloader, threshold, mask_only=True, soft=False, device='cuda'):
    fig, axes = plot_examples(model, dataloader, threshold=threshold, mask_only=mask_only, soft=soft)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    
    logger.experiment.log({'Validation Examples (Opt threshold)': wandb.Image(img)})
    plt.close(fig)