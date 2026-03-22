import io
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


class DINOv3MLPModel(nn.Module):
    def __init__(self, model_name="facebook/dinov3-vitb16-pretrain-lvd1689m", num_vpt=50):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.embed_dim = self.backbone.config.hidden_size
        self.num_vpt = num_vpt
        self.num_layers = len(self.backbone.encoder.layer)

        self.vpt_embeddings = nn.Parameter(torch.randn(self.num_layers, num_vpt, self.embed_dim))

        # Evenly spaced target layers — works for any backbone depth
        self.target_layers = set([
            int((i + 1) * self.num_layers / 4) - 1 for i in range(4)
        ])

        self.mlp_head = nn.Sequential(
            nn.Linear(self.embed_dim * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.backbone.embeddings(x)

        collected_features = []
        for i, layer in enumerate(self.backbone.encoder.layer):
            prompts = self.vpt_embeddings[i].unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([prompts, x], dim=1)
            x = layer(x)[0]
            x = x[:, self.num_vpt:, :]  # strip VPT tokens

            if i in self.target_layers:
                collected_features.append(x[:, 1:, :])  # skip CLS token

        combined = torch.cat(collected_features, dim=-1)  # (B, num_patches, embed_dim*4)
        logits = self.mlp_head(combined)                   # (B, num_patches, 1)

        num_patches = logits.shape[1]
        grid_size = int(num_patches ** 0.5)
        mask_low_res = logits.reshape(B, 1, grid_size, grid_size)

        return F.interpolate(mask_low_res, size=(H, W), mode='bilinear', align_corners=False)


class DINOv3MLP(pl.LightningModule):

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        num_vpt: int = 50,
        threshold: float = 0.5,
        lr: float = 1e-4,
        wd: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.wd = wd
        self.betas = (beta1, beta2)
        self.threshold = threshold

        self.model = DINOv3MLPModel(model_name=model_name, num_vpt=num_vpt)

        self.sigmoid = nn.Sigmoid()
        self.bce_loss = smp.losses.FocalLoss(mode='binary')
        self.dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)

    def _forward_pass(self, batch):
        imgs, targets = batch
        y_hat = self.model(imgs)
        loss = self.bce_loss(y_hat, targets) + self.dice_loss(y_hat, targets)
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
        imgs, targets = batch
        y_hat = self.model(imgs)
        loss = self.bce_loss(y_hat, targets) + self.dice_loss(y_hat, targets)
        y_pred = self.sigmoid(y_hat)
        dice_loss = dice_coef(targets, y_pred, thr=self.threshold)
        self.log('test/loss', loss,      on_step=False, on_epoch=True)
        self.log('test/dice', dice_loss, on_step=False, on_epoch=True)
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
            filter(lambda p: p.requires_grad, self.model.parameters()),
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
