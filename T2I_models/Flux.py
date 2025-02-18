from diffusers import FluxPipeline
import torch
from torch import nn
from transformers import BitsAndBytesConfig


class Flux(nn.Module):
    def __init__(self, args):
        super(Flux, self).__init__()
        self.device = args.device
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        self.pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",
                                                     device_map='balanced')
        # self.pipeline = self.pipeline.to(device=args.device)

    def forward(self, prompt):
        image = self.pipeline(prompt,
                              ).images[0]
        return image
