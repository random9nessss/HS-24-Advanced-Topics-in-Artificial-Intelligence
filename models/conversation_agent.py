import torch
import logging
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils.utils import get_device
from transformers.utils.logging import disable_progress_bar

logging.getLogger("transformers").setLevel(logging.ERROR)
disable_progress_bar()

class ConversationAgent:
    def __init__(self, model_name="google/flan-t5-xl", max_length=150):
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
            temperature=0.8,
        )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
