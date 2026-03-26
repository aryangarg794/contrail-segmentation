import io
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import segmentation_models_pytorch as smp
import wandb

from PIL import Image
from transformers import AutoModel, get_cosine_schedule_with_warmup

from contrail_segmentation.data.plotting import plot_examples
from contrail_segmentation.data.utils import TEST_IDXS
from contrail_segmentation.train.utils import dice_coef


class DINOv2ProbeModel(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.embed_dim = self.backbone.config.hidden_size
        self.num_layers = len(self.backbone.encoder.layer)

        # Evenly spaced target layers — same strategy as VPT version
        self.target_layer_indices = [
            int((i + 1) * self.num_layers / 4) - 1 for i in range(4)
        ]

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim * 4),  # DINO intermediate features are not normalized
            nn.Linear(self.embed_dim * 4, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

        # init final bias to the contrail prior (~2% positive pixels)
        # so the model starts at p≈0.02 everywhere instead of p=0.5
        prior = 0.02
        self.mlp_head[-1].bias.data.fill_(math.log(prior / (1 - prior)))  # mlp_head[-1] is still the final Linear

    def forward(self, x):
        B, _, H, W = x.shape

        outputs = self.backbone(x, output_hidden_states=True)
        # hidden_states[0] = patch embeddings, [1:] = encoder layer outputs
        hidden_states = outputs.hidden_states

        collected = []
        for layer_idx in self.target_layer_indices:
            feat = hidden_states[layer_idx + 1]  # (B, num_patches+1, embed_dim)
            collected.append(feat[:, 1:, :])      # skip CLS token

        combined = torch.cat(collected, dim=-1)   # (B, num_patches, embed_dim*4)
        logits = self.mlp_head(combined)           # (B, num_patches, 1)

        num_patches = logits.shape[1]
        grid_size = int(num_patches ** 0.5)
        mask_low_res = logits.reshape(B, 1, grid_size, grid_size)

        return F.interpolate(mask_low_res, size=(H, W), mode='bilinear', align_corners=False)


class DINOv2Probe(pl.LightningModule):

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        threshold: float = 0.5,
        lr: float = 1e-4,
        wd: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        pos_weight: float = 50.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.wd = wd
        self.betas = (beta1, beta2)
        self.threshold = threshold

        self.model = DINOv2ProbeModel(model_name=model_name)

        self.sigmoid = nn.Sigmoid()
        self.bce_loss     = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.tversky_loss = smp.losses.TverskyLoss(mode='binary', from_logits=True,
                                                    alpha=tversky_alpha, beta=tversky_beta)

    def _forward_pass(self, batch):
        imgs, targets = batch
        y_hat = self.model(imgs)
        loss = 0.3 * self.bce_loss(y_hat, targets.float()) + 0.7 * self.tversky_loss(y_hat, targets)
        dice = dice_coef(targets, y_hat.detach(), thr=self.threshold)
        return loss, dice

    def training_step(self, batch, batch_idx):
        loss, dice = self._forward_pass(batch)
        self.log('train/loss', loss, on_step=True,  on_epoch=True, prog_bar=True)
        self.log('train/dice', dice, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, dice = self._forward_pass(batch)
        self.log('val/loss', loss, on_step=True,  on_epoch=True, prog_bar=True)
        self.log('val/dice', dice, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, dice = self._forward_pass(batch)
        self.log('test/loss', loss, on_step=False, on_epoch=True)
        self.log('test/dice', dice, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        fig, axes = plot_examples(self, idxs=TEST_IDXS, mask_only=self.mask_only)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        self.logger.experiment.log({'Validation Examples': wandb.Image(img)})
        import matplotlib.pyplot as plt
        plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.mlp_head.parameters(),
            lr=self.lr,
            weight_decay=self.wd,
            betas=self.betas,
        )
        total_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(0.05 * total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
