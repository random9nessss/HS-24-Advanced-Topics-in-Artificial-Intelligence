import string
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from rapidfuzz import fuzz
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrowdSourcing:
    def __init__(self, correction_index_path, qa_pairs_path, recommender):
        """
        Args:
            correction_index (dict): Entities and corrections.
            qa_pairs (list): Q&A pairs with 'question' and 'answer'.
            recommender: Extracts entities from query.
        """
        with open(correction_index_path, 'r', encoding='utf-8') as f:
            correction_index_loaded = json.load(f)

        for key, val in correction_index_loaded.items():
            correction_index_loaded[key] = [tuple(item) for item in val]

        with open(qa_pairs_path, 'r', encoding='utf-8') as f:
            qa_pairs_loaded = json.load(f)

        self.correction_index = correction_index_loaded
        self.qa_pairs = qa_pairs_loaded
        self.recommender = recommender

        self.known_entities = {entity.lower() for entity in self.correction_index.keys()}

        # Build a comprehensive vocabulary of known tokens from QA pairs and correction_index
        self.known_tokens = self.build_known_tokens(self.correction_index, self.qa_pairs)

        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def build_known_tokens(self, correction_index, qa_pairs):
        vocab = set()

        for entity, corrections in correction_index.items():
            vocab.update(entity.lower().split())
            for c in corrections:
                orig_pred, orig_obj, corr_pred, corr_obj, _, _, _ = c
                if orig_pred: vocab.update(orig_pred.lower().split())
                if orig_obj: vocab.update(orig_obj.lower().split())
                if corr_pred: vocab.update(corr_pred.lower().split())
                if corr_obj: vocab.update(corr_obj.lower().split())

        for qa in qa_pairs:
            for w in qa['question'].lower().translate(str.maketrans('', '', string.punctuation)).split():
                vocab.add(w)

        vocab = {v for v in vocab if v.strip()}
        return vocab

    def robust_token_correction(self, text):
        """
        Correct each token by fuzzy matching against self.known_tokens.
        """
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)

        corrected_tokens = []
        for token in tokens:
            if token.isalpha():
                best_match = None
                best_score = 0
                for term in self.known_tokens:
                    score = fuzz.ratio(token, term)
                    if score > best_score:
                        best_score = score
                        best_match = term

                if best_score > 75:
                    corrected_tokens.append(best_match)
                else:
                    corrected_tokens.append(token)
            else:
                if token.isalnum():
                    corrected_tokens.append(token)

        return " ".join(corrected_tokens)


    def normalize(self, text):
        # Tokenize after correction
        tokens = word_tokenize(text)
        filtered_tokens = []
        for token in tokens:
            if token.isalpha() and token not in self.stopwords:
                lemma = self.lemmatizer.lemmatize(token)
                filtered_tokens.append(lemma)
        return " ".join(filtered_tokens)


    def classify_query(self, query):
        extracted_entities = self.recommender.extract_entities(query)
        logger.info(extracted_entities)

        if not extracted_entities:
            return "factual"

        all_extracted = set()
        for ents in extracted_entities.values():
            all_extracted.update(ents)

        recognized_entity = None
        for e in all_extracted:
            if e.lower() in self.known_entities:
                recognized_entity = e
                break

        if not recognized_entity:
            return "factual"

        corrected_query = self.robust_token_correction(query)
        user_query_norm = self.normalize(corrected_query)
        user_embedding = self.model.encode([user_query_norm], convert_to_tensor=True)

        qa_questions_norm = [self.normalize(self.robust_token_correction(p['question'])) for p in self.qa_pairs]
        qa_embeddings = self.model.encode(qa_questions_norm, convert_to_tensor=True)

        cos_scores = util.pytorch_cos_sim(user_embedding, qa_embeddings).squeeze()
        max_score = float(cos_scores.max().cpu().item())

        logger.info(max_score)

        threshold = 0.75
        if max_score > threshold:
            best_idx = int(np.argmax(cos_scores.cpu().numpy()))
            return self.qa_pairs[best_idx]['answer']
        else:
            return "factual"