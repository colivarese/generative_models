import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer




class CLIPTextEncoder(nn.Module):
    def __init__(self, pretrained_model_name: str = "openai/clip-vit-base-patch16", device="cuda"):
        super(CLIPTextEncoder, self).__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name)
        self.model = CLIPTextModel.from_pretrained(pretrained_model_name)
        self.model.to(device)
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def forward(self, input_text):
        tokens = self.tokenizer(input_text, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        tokens = tokens.to(self.device)

        outputs = self.model(**tokens)
        text_embeddings = outputs.last_hidden_state
        return text_embeddings
    

