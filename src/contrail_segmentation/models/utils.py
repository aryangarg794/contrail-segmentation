import torch

from segmentation_models_pytorch.metrics import get_stats, iou_score, f1_score 
from torchmetrics.functional import auroc

@torch.no_grad()
def compute_metrics(y_hat, targets, thr=0.5):
    probs = torch.sigmoid(y_hat)
    preds = (probs > thr).long()
    targets = targets.long()

    tp, fp, fn, tn = get_stats(preds, targets, mode="binary")

    metrics = {
        "iou": iou_score(tp, fp, fn, tn, reduction="micro"),
        "f1": f1_score(tp, fp, fn, tn, reduction="micro"),
        "recall": tp / (tp + fn + 1e-7),
        "precision": tp / (tp + fp + 1e-7),
        "specificity": tn / (tn + fp + 1e-7),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + 1e-7),
        "auc": auroc(probs, targets, task="binary", thresholds=200)
    }

    metrics = {k: v.mean() for k, v in metrics.items()}
    return metrics