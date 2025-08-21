import torch
from torch import nn
from diffusers import StableDiffusion3Pipeline


class SD3(nn.Module):
    def __init__(self, args):
        super(SD3, self).__init__()
        self.pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16).to(device=args.device)

    def forward(self, prompt):
        image = self.pipe(
                        prompt,
                        num_inference_steps=28,
                        guidance_scale=7.0,
                    ).images[0]
        return image