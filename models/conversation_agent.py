import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils.utils import get_device

class ConversationAgent:
    def __init__(self, model_name="google/flan-t5-large", max_length=150):
        self.device = get_device()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.max_length = max_length

    def generate_response(self, prompt):
        """
        Generates a response based on the given prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=self.max_length,
            num_beams=5,
            early_stopping=True,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
