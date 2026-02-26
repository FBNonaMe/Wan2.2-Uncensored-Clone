import torch
from tqdm.auto import tqdm
from models.text_encoder import TextEncoder
from models.wan_transformer import WanT2V
from models.wan_vae import WanVideoVAE

class WanFlowMatchingPipeline:
    """
    Real Wan 2.1/2.2 Pipeline using Flow Matching
    """
    def __init__(self, device="cuda"):
        self.device = device
        self.text_encoder = TextEncoder().to(device)
        self.model = WanT2V().to(device)
        self.vae = WanVideoVAE().to(device)

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        num_frames=16,
        height=480,
        width=832,
        num_steps=50,
        guidance_scale=5.0,
        seed=None
    ):
        if seed is not None:
            torch.manual_seed(seed)

        # 1. Encode Text
        context = self.text_encoder([prompt])
        null_context = self.text_encoder([""])
        
        # 2. Initial Noise
        # Wan uses 16 channels for latents
        latents = torch.randn(
            (1, 16, num_frames, height // 8, width // 8),
            device=self.device
        )
        
        # 3. Flow Matching Sampling (Euler)
        dt = 1.0 / num_steps
        for i in tqdm(range(num_steps)):
            t = torch.ones((1,), device=self.device) * (1.0 - i * dt)
            
            # CFG
            latent_model_input = torch.cat([latents] * 2)
            t_input = torch.cat([t] * 2)
            context_input = torch.cat([null_context, context])
            
            # Predict velocity (v-prediction)
            v_pred = self.model(latent_model_input, t_input, context_input)
            
            v_uncond, v_cond = v_pred.chunk(2)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
            
            # Euler step
            latents = latents - v * dt

        # 4. Decode
        video = self.vae.decode(latents)
        video = (video / 2 + 0.5).clamp(0, 1)
        
        return video
