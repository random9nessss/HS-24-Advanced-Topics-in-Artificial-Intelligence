import torch
import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from utils.utils import get_device

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.WARNING)

class NERParser:
    def __init__(self, model_name="dslim/bert-large-NER", lowercase=False):
        self.lowercase = lowercase
        self.device = get_device()

        self.nlp_pipeline = pipeline(
            "ner",
            model=AutoModelForTokenClassification.from_pretrained(model_name),
            tokenizer=AutoTokenizer.from_pretrained(model_name, do_lower_case=lowercase),
            device=self.device,
            aggregation_strategy="simple"
        )

    def parse_ner_results(self, ner_results):
        per_entities = [e['word'] for e in ner_results if e['entity_group'] == 'PER']
        misc_entities = [e['word'] for e in ner_results if e['entity_group'] == 'MISC']
        return per_entities, misc_entities

    def process_query(self, query):
        if self.lowercase:
            query = query.lower()
        return self.parse_ner_results(self.nlp_pipeline(query))
