import torch
import random
import numpy as np

def set_seed(seed):
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    return seed
