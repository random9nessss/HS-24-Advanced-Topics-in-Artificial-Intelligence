import torch
import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from flair.data import Sentence
from flair.models import SequenceTagger
from utils.utils import get_device
import difflib

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("flair").setLevel(logging.WARNING)

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

        self.flair_tagger = SequenceTagger.load("flair/ner-english")

    def parse_ner_results(self, ner_results):
        """Parse results from a Hugging Face NER model."""
        per_entities = [e['word'] for e in ner_results if e['entity_group'] == 'PER']
        misc_entities = [e['word'] for e in ner_results if e['entity_group'] == 'MISC']
        return per_entities, misc_entities

    def parse_flair_results(self, sentence):
        """Parse results from the Flair NER model."""
        per_entities = []
        misc_entities = []
        for entity in sentence.get_spans('ner'):
            if entity.get_label("ner").value == "PER":
                per_entities.append(entity.text)
            elif entity.get_label("ner").value == "MISC":
                misc_entities.append(entity.text)
        return per_entities, misc_entities


    def process_query(self, query):
        """Process the query using both BERT-based and Flair NER models, plus gazetteer matching."""
        if self.lowercase:
            query = query.lower()

        # Primary NER model
        ner_results_bert = self.nlp_pipeline(query)
        per_entities_bert, misc_entities_bert = self.parse_ner_results(ner_results_bert)

        # Flair NER model (particularly good for persons)
        sentence = Sentence(query)
        self.flair_tagger.predict(sentence)
        per_entities_flair, misc_entities_flair = self.parse_flair_results(sentence)

        per_entities = list(set(per_entities_bert + per_entities_flair))
        misc_entities = list(set(misc_entities_bert + misc_entities_flair))

        return per_entities, misc_entities
