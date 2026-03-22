import numpy as np
import torch
import torch.nn.functional as F

from rich.progress import (
    BarColumn, 
    MofNCompleteColumn, 
    Progress, 
    TextColumn, 
    TimeElapsedColumn
)

def dice_coef(y_true, y_pred, thr=0.5, epsilon=1e-7, pos_only=False):
    y_pred = F.sigmoid(y_pred)
    if thr is not None:
        y_pred = (y_pred > thr).float()
    
    y_true_flat = y_true.view(y_true.size(0), -1)
    y_pred_flat = y_pred.view(y_pred.size(0), -1)

    intersection = (y_true_flat * y_pred_flat).sum(dim=1)
    denominator = y_true_flat.sum(dim=1) + y_pred_flat.sum(dim=1)
    
    dice_per_sample = (2.0 * intersection + epsilon) / (denominator + epsilon)

    if pos_only:
        is_positive_sample = (y_true_flat.sum(dim=1) > 0).float()
        num_positive_samples = is_positive_sample.sum()
        if num_positive_samples > 0:
            return (dice_per_sample * is_positive_sample).sum() / num_positive_samples
        else:
            return torch.tensor(0.0, device=y_true.device)
    else:
        return dice_per_sample.mean()


@torch.no_grad()
def find_best_threshold(model, dataloader, num_vals=100, device='cuda'):
    model.eval()
    all_preds = []
    all_targets = []
    for batch in dataloader:
        imgs, target = batch
        imgs.to(device=device)
        target.to(device=device)
        
        logits = model.model(imgs)
        all_preds.append(logits.detach().cpu())
        all_targets.append(target.cpu())
        
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    
    progress_bar = Progress(
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
    )
    
    best = -1
    best_thr = None
    
    thresholds = np.linspace(0, 1, num_vals)
    with progress_bar as p:
        for thr in thresholds:
            dice = dice_coef(targets, preds, thr)
            if dice > best:
                best = dice
                best_thr = thr
                
    return best_thr