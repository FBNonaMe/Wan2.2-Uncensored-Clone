import cv2
import numpy as np
import torch

def save_video(video_tensor, output_path, fps=8):
    # video_tensor: (B, T, C, H, W) in [0, 1]
    video = video_tensor[0].cpu().numpy() # (T, C, H, W)
    video = (video * 255).astype(np.uint8)
    video = video.transpose(0, 2, 3, 1) # (T, H, W, C)
    
    T, H, W, C = video.shape
    
    # Use OpenCV to write MP4
    # Note: FourCC 'mp4v' is widely supported
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    for i in range(T):
        frame = video[i]
        # RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
        
    out.release()
    return output_path
