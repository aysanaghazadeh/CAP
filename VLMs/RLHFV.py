from transformers import pipeline
import torch


class RLHFV(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.pipe = pipeline("text-generation",
                             model="openbmb/RLHF-V",
                             device_map='auto')

    def forward(self, image, prompt, generate_kwargs={"max_new_tokens": 250}):
        output = self.pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 250})
        return output