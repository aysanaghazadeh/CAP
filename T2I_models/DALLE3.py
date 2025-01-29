from torch import nn
from openai import OpenAI
import os
import requests
from PIL import Image
import io


class DALLE3(nn.Module):
    def __init__(self, args):
        super(DALLE3, self).__init__()
        os.environ["OPENAI_API_KEY"] = args.api_key
        self.client = OpenAI()

    def forward(self, prompt):
        response = self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            quality="standard",
            size="1024x1024",
            n=1,
        )
        image_url = response.data[0].url
        # Download the image
        response = requests.get(image_url)
        image_bytes = io.BytesIO(response.content)

        # Open the image with PIL
        image = Image.open(image_bytes)
        return image
