import numpy as np
import torch
import torch.nn as nn


class SoftHoughTransform(nn.Module):
    """Differentiable soft Hough transform via bilinear scatter."""

    def __init__(self, H: int, W: int, num_angles: int = 90):
        super().__init__()
        self.H = H
        self.W = W
        self.num_angles = num_angles

        thetas = torch.linspace(0, np.pi * (1 - 1 / num_angles), num_angles)

        max_rho = int(np.ceil(np.sqrt(H ** 2 + W ** 2)))
        self.num_rhos = 2 * max_rho + 1
        rho_min = float(-max_rho)
        rho_max = float(max_rho)

        # Pixel coords centred at image centre
        ys = torch.arange(H).float() - H / 2
        xs = torch.arange(W).float() - W / 2
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # [H, W]

        # rho_map[a, h, w] = x*cos(t) + y*sin(t), normalised to [0, num_rhos-1]
        rho_map = (
            xx.unsqueeze(0) * torch.cos(thetas).view(-1, 1, 1)
            + yy.unsqueeze(0) * torch.sin(thetas).view(-1, 1, 1)
        )  # [A, H, W]
        rho_norm = (rho_map - rho_min) / (rho_max - rho_min) * (self.num_rhos - 1)

        rho_norm_flat = rho_norm.reshape(num_angles, -1)  # [A, H*W]
        rho_floor = rho_norm_flat.long().clamp(0, self.num_rhos - 2)
        rho_frac = rho_norm_flat - rho_floor.float()
        rho_ceil = (rho_floor + 1).clamp(0, self.num_rhos - 1)

        self.register_buffer('rho_floor', rho_floor)   # [A, H*W]
        self.register_buffer('rho_frac',  rho_frac)    # [A, H*W]
        self.register_buffer('rho_ceil',  rho_ceil)    # [A, H*W]

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mask: [B, 1, H, W] — float probabilities in [0, 1]
        Returns:
            hough: [B, num_angles, num_rhos]
        """
        B = mask.shape[0]
        mask_flat = mask.squeeze(1).reshape(B, -1).float()  # always fp32

        hough = torch.zeros(
            B, self.num_angles, self.num_rhos,
            device=mask.device, dtype=torch.float32,
        )
        for a in range(self.num_angles):
            w_floor = (1 - self.rho_frac[a]) * mask_flat   # [B, H*W]
            w_ceil  =      self.rho_frac[a]  * mask_flat   # [B, H*W]
            idx_floor = self.rho_floor[a].unsqueeze(0).expand(B, -1)
            idx_ceil  = self.rho_ceil[a].unsqueeze(0).expand(B, -1)
            hough[:, a, :].scatter_add_(1, idx_floor, w_floor)
            hough[:, a, :].scatter_add_(1, idx_ceil,  w_ceil)

        return hough  # [B, A, num_rhos]


def _soft_dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_flat   = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    inter = (pred_flat * target_flat).sum(dim=1)
    den   = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    return (1 - (2 * inter + eps) / (den + eps)).mean()


class SRLoss(nn.Module):
    """
    SR Loss from Sun & Roosenbrand (2025):
      L_SR = (1 - alpha) * L_Dice(p, g) + alpha * L_Dice(p_h, g_h)

    Combines pixel-space Dice with Hough-space Dice to exploit the linear
    shape of contrails.

    Args:
        H, W:       spatial size of the segmentation output (default 256)
        num_angles: Hough angle discretisation (default 90)
        alpha:      weight of Hough-space term (default 0.5)
    """

    def __init__(self, H: int = 256, W: int = 256, num_angles: int = 90, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.hough = SoftHoughTransform(H, W, num_angles)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   logits  [B, 1, H, W]
            target: binary mask [B, 1, H, W]
        """
        pred_prob = torch.sigmoid(pred)
        target_f  = target.float()

        loss_image = _soft_dice_loss(pred_prob, target_f)

        pred_h   = self.hough(pred_prob)   # [B, A, num_rhos]
        target_h = self.hough(target_f)    # [B, A, num_rhos]

        # Normalise per sample so values are in [0, 1]
        pred_h   = pred_h   / (pred_h.amax(dim=[1, 2], keepdim=True)   + 1e-8)
        target_h = target_h / (target_h.amax(dim=[1, 2], keepdim=True) + 1e-8)

        loss_hough = _soft_dice_loss(pred_h, target_h)

        return (1 - self.alpha) * loss_image + self.alpha * loss_hough
