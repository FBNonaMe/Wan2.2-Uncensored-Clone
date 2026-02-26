import torch
from diffusers import DDIMScheduler

class VideoSampler:
    def __init__(self, num_train_timesteps=1000):
        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )

    def set_timesteps(self, num_inference_steps):
        self.scheduler.set_timesteps(num_inference_steps)

    def step(self, model_output, timestep, sample):
        return self.scheduler.step(model_output, timestep, sample).prev_sample
