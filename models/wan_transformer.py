import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class WanAttention(nn.Module):
    def __init__(self, dim, num_heads=16, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rope=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if rope is not None:
            # Apply 3D RoPE (Simplified implementation)
            pass

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class WanTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WanAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        # Adaptive Layer Norm (adaLN-Single) for conditioning
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(self, x, c):
        # c is the conditioning embedding (time + text)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # MSA path
        h = x
        x = self.norm1(x)
        x = x * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = self.attn(x)
        x = h + gate_msa.unsqueeze(1) * x
        
        # MLP path
        h = x
        x = self.norm2(x)
        x = x * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = self.mlp(x)
        x = h + gate_mlp.unsqueeze(1) * x
        
        return x

class WanT2V(nn.Module):
    """
    Wan 2.1/2.2 Transformer (DiT) Architecture
    """
    def __init__(self, input_dim=16, dim=1152, num_heads=16, num_layers=28):
        super().__init__()
        self.patch_embed = nn.Linear(input_dim, dim)
        
        self.blocks = nn.ModuleList([
            WanTransformerBlock(dim, num_heads) for _ in range(num_layers)
        ])
        
        self.final_layer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, input_dim)
        )
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x, t, context):
        # x: (B, C, T, H, W) - Latents
        # t: (B,) - Timesteps
        # context: (B, N, D) - Text embeddings
        
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> b (t h w) c')
        x = self.patch_embed(x)
        
        # Simplified conditioning: combine time and pooled text
        t_emb = self.time_embed(t)
        c = t_emb + context.mean(dim=1)
        
        for block in self.blocks:
            x = block(x, c)
            
        x = self.final_layer(x)
        x = rearrange(x, 'b (t h w) c -> b c t h w', t=T, h=H, w=W)
        return x
