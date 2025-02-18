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
                                                     torch_dtype=torch.float16,
                                                     quantization_config=quantization_config)

    def forward(self, prompt):
        image = self.pipeline(prompt,
                              height=512,
                              width=512,
                              guidance_scale=3.5,
                              num_inference_steps=28,
                              max_sequence_length=512
                              ).images[0]
        return image
