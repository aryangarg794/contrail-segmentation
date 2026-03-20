import torch
import torch.nn as nn
from transformers import AutoModel

class DINOv3_MLP(nn.Module):
    def __init__(self, model_name="facebook/dinov3-vitl16", num_vpt=50):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.embed_dim = self.backbone.config.hidden_size
        self.num_vpt = num_vpt
        self.num_layers = len(self.backbone.encoder.layer)
        
        self.vpt_embeddings = nn.Parameter(torch.randn(self.num_layers, num_vpt, self.embed_dim))
        
        self.mlp_head = nn.Sequential(
            nn.Linear(self.embed_dim * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.backbone.embeddings(x)
        
        collected_features = []
        target_layers = [5, 11, 17, 23] 
        
        for i, layer in enumerate(self.backbone.encoder.layer):
            prompts = self.vpt_embeddings[i].expand(batch_size, -1, -1)
            x = torch.cat((prompts, x), dim=1)
            x = layer(x)[0]
            x = x[:, self.num_vpt:, :]
            
            if i in target_layers:
                collected_features.append(x[:, 5:, :])

        combined = torch.cat(collected_features, dim=-1)
        logits = self.mlp_head(combined)
        
        grid_size = int(logits.shape[1]**0.5)
        mask_low_res = logits.reshape(batch_size, 1, grid_size, grid_size)
        
        return torch.sigmoid(nn.functional.interpolate(mask_low_res, scale_factor=16, mode='bilinear'))