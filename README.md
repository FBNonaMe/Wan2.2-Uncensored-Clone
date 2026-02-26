# WanVideo-Clone-T2V

A high-performance Text-to-Video diffusion system inspired by Wan 2.x, designed for user-space execution and ComfyUI integration.

## Architecture
- **Text Encoder**: T5-v1.1-base for rich semantic understanding.
- **Temporal UNet**: 2D UNet backbone with temporal attention layers for coherent motion.
- **Video VAE**: Latent space compression with temporal frame handling.
- **Pipeline**: DDIM-based sampling in 5D latent space (B, C, T, H, W).

## Installation
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux
   venv\Scripts\activate     # Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage (Python)
```python
from pipeline.video_pipeline import WanVideoPipeline
from utils.video_writer import save_video

pipe = WanVideoPipeline(device="cuda")
video = pipe("A cinematic shot of a dragon flying over a volcano", num_frames=16)
save_video(video, "dragon.mp4", fps=8)
```

## ComfyUI Integration
1. Copy the `comfyui_wan_clone` folder to your `ComfyUI/custom_nodes/` directory.
2. Restart ComfyUI.
3. Find the node under `WanVideo -> Wan Video Generator (T2V)`.

## Requirements
- Python 3.10+
- PyTorch 2.0+
- NVIDIA GPU with 16GB+ VRAM (recommended for inference)
