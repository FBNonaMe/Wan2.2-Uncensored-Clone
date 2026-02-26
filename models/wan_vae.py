import torch
import torch.nn as nn
from einops import rearrange

class CausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # Causal convolution only looks at past frames
        self.pad = (kernel_size[2] - 1, 0, kernel_size[1] // 2, kernel_size[1] // 2, kernel_size[0] // 2, kernel_size[0] // 2)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=0)

    def forward(self, x):
        x = F.pad(x, self.pad)
        return self.conv(x)

class WanVideoVAE(nn.Module):
    """
    Wan 2.1/2.2 Causal 3D VAE
    """
    def __init__(self, latent_dim=16):
        super().__init__()
        # This is a highly simplified structure of the Causal 3D VAE
        # In reality, it has multiple residual blocks and temporal downsampling
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(128, latent_dim, kernel_size=3, padding=1)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv3d(latent_dim, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(128, 3, kernel_size=3, padding=1)
        )

    def encode(self, x):
        # x: (B, T, C, H, W)
        x = rearrange(x, 'b t c h w -> b c t h w')
        z = self.encoder(x)
        return z

    def decode(self, z):
        # z: (B, C, T, H, W)
        x = self.decoder(z)
        x = rearrange(x, 'b c t h w -> b t c h w')
        return x
