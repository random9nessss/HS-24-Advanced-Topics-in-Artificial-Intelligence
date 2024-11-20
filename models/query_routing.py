import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

class QueryRouter:
    def __init__(self):
        """
        Initializes the QuestionClassifier with a pre-trained model and tokenizer.

        Args:
            model_path (str): Path to the directory where the fine-tuned model and tokenizer are saved.
        """
        model_path = os.path.join(os.path.dirname(__file__), '../models/question_classifier_model')
        model_path = os.path.abspath(model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # Mapping from label indices to question types
        self.label_map = {
            0: 'factual',
            1: 'recommendation',
            2: 'multimedia',
            3: 'unrelated'
        }

    def predict(self, query):
        """
        Classifies a single question into one of the predefined categories.

        Args:
            question (str): The input question to classify.

        Returns:
            str: The predicted category label as a string.
        """
        # Tokenization and Encoding of Query
        inputs = self.tokenizer.encode_plus(
            query.lower(),
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        # Prediction
        outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        return self.label_map[predicted_class]