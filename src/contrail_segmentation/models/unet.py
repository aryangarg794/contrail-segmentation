import torch
import numpy as np 
import lightning as pl
import torch.nn as nn

from torch import Tensor
from typing import Self, List

# implementation loosely inspired by https://medium.com/data-science/diffusion-model-from-scratch-in-pytorch-ddpm-9d9760528946

class Sinusoidal(nn.Module):
    
    def __init__(
        self: Self,
        embed_size: int, 
        horizon: int = 1000,
        *args, 
        **kwargs
    ) -> None:
        super(Sinusoidal, self).__init__(*args, **kwargs) 
        
        pe = torch.zeros(horizon, embed_size, requires_grad=False)
        positions = torch.arange(0, horizon).unsqueeze(dim=1)
        div = torch.exp(torch.arange(0, embed_size, 2).float() * -(np.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(positions * div)
        pe[:, 1::2] = torch.cos(positions * div)
        
        self.embed_size = embed_size
        self.register_buffer("pe", pe) 

    def forward(self: Self, t: int) -> Tensor:
        return self.pe[t].view(-1, self.embed_size, 1, 1)
    
    
class ResidualBlock(nn.Module):
    
    def __init__(
        self: Self,
        in_channels: int,
        horizon: int = 1000, 
        dropout: float = 0.1, 
        activation: nn.Module = nn.ReLU,
        groups: int = 32,
        *args, 
        **kwargs
    ) -> None:
        super(ResidualBlock, self).__init__(*args, **kwargs)
        
        self.sinusoidal = Sinusoidal(in_channels, horizon)
        
        self.layers = nn.Sequential(
            nn.GroupNorm(num_groups=groups, num_channels=in_channels),
            activation(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
            nn.Dropout(p=dropout),
            nn.GroupNorm(num_groups=groups, num_channels=in_channels),
            activation(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        )
        
    def forward(self: Self, x: Tensor, t: int, embed: bool) -> Tensor:
        if embed: 
            x = x + self.sinusoidal(t)
        return self.layers(x) + x * 1/np.sqrt(2)
    
    
class UNETLayer(nn.Module):
    
    def __init__(
        self: Self,
        in_channels: int, 
        out_channels: int, 
        resblock: ResidualBlock = ResidualBlock,
        upsample: bool = False, 
        attention: bool = False, 
        num_heads: int = 8, 
        dropout: float = 0.1,
        *args, 
        **kwargs
    ) -> None:
        super(UNETLayer, self).__init__(*args, **kwargs)
        
        self.resblock1 = resblock(in_channels=in_channels)    
        self.resblock2 = resblock(in_channels=in_channels)
        
        if upsample:
            self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, 
                                           kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=3, stride=2, padding=1)
            
        self.attention = attention
        if attention: 
            self.attention_layer = nn.MultiheadAttention(
                embed_dim=in_channels, 
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        
    def forward(self: Self, x: Tensor, t: int) -> None: 
        x = self.resblock1(x, t, True)
        if self.attention:
            batch_size, channels, height, width = x.shape
            x = x.view(batch_size, channels, -1).transpose(1, 2) # attention on patches
            x, _ = self.attention_layer(x, x, x)
            x = x.transpose(1, 2).view(batch_size, channels, height, width)
    
        x = self.resblock2(x, t, False)
        return self.conv(x), x
    
    
class UNETBase(nn.Module):
    
    def __init__(
        self: Self,
        in_channels: int = 3, 
        channels: List = list([64, 128, 256, 512, 512, 384, 192]),
        *args, 
        **kwargs
    ) -> None:
        super(UNETBase, self).__init__(*args, **kwargs)
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.layer1 = UNETLayer(in_channels=64, out_channels=128) # 64 -> 128 
        self.layer2 = UNETLayer(in_channels=128, out_channels=256, attention=True) # 128 -> 256
        self.layer3 = UNETLayer(in_channels=256, out_channels=512) # 256 -> 512
        self.layer4 = UNETLayer(in_channels=512, out_channels=256, upsample=True) # 512 -> 256
        self.layer5 = UNETLayer(in_channels=512, out_channels=256, upsample=True) # 512 -> 384
        self.layer6 = UNETLayer(in_channels=384, out_channels=128, attention=True, upsample=True) # 384 -> 192 
        
        self.conv2 = nn.Conv2d(in_channels=channels[6], out_channels=channels[6]//2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=channels[6]//2, out_channels=in_channels, kernel_size=1)
        self.relu = nn.ReLU()
        
    def forward(self: Self, x: Tensor, t: int) -> Tensor:
        x = self.conv1(x)
        x, x1 = self.layer1(x, t)
        x, x2 = self.layer2(x, t)
        x, x3 = self.layer3(x, t)

        x, _ = self.layer4(x, t)
        x, _ = self.layer5(torch.cat([x3, x], dim=1), t)
        x, _ = self.layer6(torch.cat([x2, x], dim=1), t)
        x = self.conv2(torch.cat([x1, x], dim=1))
        x = self.relu(x)
        x = self.conv3(x)
        
        return x

        
class UNET(pl.LightningModule):
    
    def __init__(
        self, 
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
    