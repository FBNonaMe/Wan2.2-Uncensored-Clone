import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Tokenizer

class TextEncoder(nn.Module):
    def __init__(self, model_id="google/t5-v1_1-base"):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_id)
        self.encoder = T5EncoderModel.from_pretrained(model_id)

    def forward(self, prompt):
        # prompt: List[str]
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}
        
        outputs = self.encoder(**inputs)
        last_hidden_state = outputs.last_hidden_state
        
        return last_hidden_state
