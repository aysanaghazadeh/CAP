from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import PeftModel
import os


class LLAMA3(nn.Module):
    def __init__(self, args):
        super(LLAMA3, self).__init__()
        self.args = args
        if not args.train:
            # device_map = {
            #     "language_model": 1,
            #     "language_projection": 2
            # }
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B",
                                                           token=os.environ.get('HF_TOKEN'))
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B",
                                                              token=os.environ.get('HF_TOKEN'),
                                                              device_map="auto",
                                                              quantization_config=quantization_config)
            if args.fine_tuned:
                self.model = PeftModel.from_pretrained(self.model, os.path.join(args.model_path, 'my_LLAMA3_large_sample_model/checkpoint-4350/'))
        

    def forward(self, inputs):
        if not self.args.train:
            inputs = self.tokenizer(inputs, return_tensors="pt").to(device=self.args.device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=250)
            output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            output = output.split(':')[-1]
            print(output)
            return output
