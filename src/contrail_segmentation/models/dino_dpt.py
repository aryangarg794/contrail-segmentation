import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class DINOv2VPT(nn.Module):
    def __init__(self, model_name="facebook/dinov2-large", num_tokens=50):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(model_name)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.num_layers = len(self.backbone.encoder.layer)
        self.embed_dim = self.backbone.config.hidden_size
        self.num_tokens = num_tokens

        self.vpt_embeddings = nn.Parameter(torch.randn(self.num_layers, self.num_tokens, self.embed_dim))

        self.projects = nn.ModuleList([nn.Conv2d(self.embed_dim, 256, kernel_size=1) for _ in range(4)])

        self.upsample = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.Upsample(scale_factor=2))
            for _ in range(3)
        ])

        self.final_conv = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]
        x = self.backbone.embeddings(x)  # (B, 1+num_patches, embed_dim)

        features = []
        target_layers = {5, 11, 17, 23}

        for i, layer in enumerate(self.backbone.encoder.layer):
            prompts = self.vpt_embeddings[i].unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([prompts, x], dim=1)
            x = layer(x)[0]
            x = x[:, self.num_tokens:, :]  # strip VPT tokens

            if i in target_layers:
                # skip CLS token (and register tokens if any); patch tokens start at index 1
                patches = x[:, 1:, :]  # (B, num_patches, embed_dim)
                h = w = int(patches.shape[1] ** 0.5)
                features.append(patches.permute(0, 2, 1).reshape(B, self.embed_dim, h, w))

        out = self.projects[3](features[3])
        for i in range(2, -1, -1):
            out = self.upsample[i](out)
            out = out + self.projects[i](features[i])

        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        return self.final_conv(out)
