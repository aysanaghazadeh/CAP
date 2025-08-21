import torch
from torch import nn
from diffusers import DiffusionPipeline


class SDXL(nn.Module):
    def __init__(self, args):
        super(SDXL, self).__init__()
        self.pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base").to(device=args.device)
        

    def forward(self, prompt):
        image = self.pipe(prompt, num_inference_steps=20).images[0]
        return image