import numpy as np
import torch

from rich.progress import (
    BarColumn, 
    MofNCompleteColumn, 
    Progress, 
    TextColumn, 
    TimeElapsedColumn
)

def dice_coef(y_true, y_pred, thr=0.5, epsilon=0.001):
    y_true = y_true.flatten()
    if thr is not None:
        y_pred = (y_pred > thr).float().flatten()
    else:
        y_pred = y_pred.flatten().float()

    inter = (y_true * y_pred).sum()
    den = y_true.sum() + y_pred.sum()
    dice = (2.0 * inter + epsilon) / (den + epsilon)
    return dice.item()


@torch.no_grad()
def find_best_threshold(model, dataloader, num_vals=100, device='cuda'):
    model.eval()
    all_preds = []
    all_targets = []
    for batch in dataloader:
        imgs, target = batch
        imgs.to(device)
        target.to(device)
        
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
    interval = 1.00 / num_vals
    with progress_bar as p:
        for thr in np.arange(0, 1 + interval, interval):
            dice = dice_coef(targets, preds, thr)
            if dice > best:
                best = dice
                best_thr = thr
                
    return best_thr