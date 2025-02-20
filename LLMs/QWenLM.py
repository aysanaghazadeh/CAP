from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn
from peft import PeftModel
import os

class QWenLM(nn.Module):
    def __init__(self, args):
        super(QWenLM, self).__init__()

        self.args = args
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        if args.fine_tuned:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            self.model = PeftModel.from_pretrained(self.model,
                                                   os.path.join(args.model_path,
                                                                'my_QWenLM_CPO/checkpoint-3000/'))
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, prompt):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=25 if self.args.fine_tuned else 512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
