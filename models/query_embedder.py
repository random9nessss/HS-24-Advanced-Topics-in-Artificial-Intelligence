import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from utils.utils import get_device
import logging

logging.getLogger("torch").setLevel(logging.WARNING)

class QueryEmbedderContextualized:
    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2'):
        """Initializes the QueryEmbedder with a SentenceTransformer model and device setup."""
        self.device = get_device()
        self.model = SentenceTransformer(model_name, device=self.device)
        self.cache = {}

    def embed_phrase(self, phrases):
        """
        Generates embeddings for given phrases using SentenceTransformer, with caching.

        Args:
            phrases (str or List[str]): Input phrase(s) to embed.

        Returns:
            np.ndarray: Embedding vector(s) for the input phrase(s).
        """
        if isinstance(phrases, str):
            phrases = [phrases]
        elif not isinstance(phrases, list):
            raise TypeError("Input must be a string or a list of strings.")

        phrases_to_compute = [p for p in phrases if p not in self.cache]
        cached_embeddings = [self.cache[p] for p in phrases if p in self.cache]

        if phrases_to_compute:
            new_embeddings = self.model.encode(
                phrases_to_compute,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            for phrase, emb in zip(phrases_to_compute, new_embeddings):
                self.cache[phrase] = emb
            cached_embeddings.extend(new_embeddings)

        return cached_embeddings[0] if len(cached_embeddings) == 1 else np.array(cached_embeddings)
