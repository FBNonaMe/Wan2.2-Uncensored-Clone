import torch
import os
from pipeline.video_pipeline import WanVideoPipeline
from utils.video_writer import save_video
from utils.seed import set_seed

class WanVideoGenerator:
    _pipeline_cache = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "frames": ("INT", {"default": 16, "min": 1, "max": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "fps": ("INT", {"default": 8, "min": 1, "max": 60}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "generate_video"
    CATEGORY = "WanVideo"

    def generate_video(self, prompt, steps, frames, seed, fps, guidance_scale):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize or retrieve cached pipeline
        if WanVideoGenerator._pipeline_cache is None:
            print(f"[WanVideo] Initializing pipeline on {device}...")
            WanVideoGenerator._pipeline_cache = WanVideoPipeline(device=device)
        else:
            print("[WanVideo] Using cached pipeline.")
        
        pipeline = WanVideoGenerator._pipeline_cache
        
        # Set seed
        set_seed(seed)
        
        # Generate
        video_tensor = pipeline(
            prompt=prompt,
            num_frames=frames,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        
        # Save to temp file
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_path = os.path.join(output_dir, f"wan_video_{seed}.mp4")
        save_video(video_tensor, output_path, fps=fps)
        
        # ComfyUI expects a specific return format for VIDEO type 
        # (usually a path or a list of frames depending on the wrapper)
        return (output_path,)

NODE_CLASS_MAPPINGS = {
    "WanVideoGenerator": WanVideoGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoGenerator": "Wan Video Generator (T2V)"
}
