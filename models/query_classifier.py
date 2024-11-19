import logging
import re
from transformers import pipeline

from utils.utils import get_device

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.WARNING)

class QueryClassifier:
    def __init__(self, model_name="facebook/bart-large-mnli"):
        """
        Initializes the query classifier with a zero-shot classification model,
        pre-compiled regex patterns, and keyword lists for efficient processing.

        Args:
            model_name (str): The name of the model to use for zero-shot classification.
        """
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=get_device()
        )

        self.intent_labels = [
            "recommendation",
            "multimedia",
            "other"
        ]

        # Pre-compile regex patterns for greetings and small talk
        greeting_patterns = [
            r'\bhello\b', r'\bhi\b', r'\bhey\b', r'\bgreetings\b', r'\bhow are you\b',
            r'\bwhat\'s up\b', r'\bhow\'s it going\b', r'\bgood morning\b', r'\bgood evening\b',
            r'\bgood afternoon\b', r'\bnice to meet you\b', r'\bpleased to meet you\b'
        ]
        self.greeting_regex = re.compile('|'.join(greeting_patterns), re.IGNORECASE)

        # List of movie-related keywords
        self.movie_keywords = [
            "movie", "film", "actor", "actress", "director", "producer", "cinema",
            "starred", "played in", "cast", "genre", "screenplay", "box office", "theater",
            "watch", "scene", "plot", "character", "award", "oscar", "hollywood", "bollywood",
            "tv show", "series", "episode", "season", "trailer", "sequel", "prequel", "franchise",
            "recommend", "suggest", "like", "similar to", "fan of", "propose", ""
        ]

        # Multimedia keywords and phrases
        self.multimedia_keywords = [
            r'picture', r'image', r'poster', r'photo', r'show', r'look',
            r'look like', r'show me', r'display', r'picture of', r'photo of'
        ]
        self.multimedia_regex = re.compile('|'.join(self.multimedia_keywords), re.IGNORECASE)

        # Recommendation keywords and phrases
        self.recommendation_keywords = [
            r'recommend', r'propose', r'advise', r'like',
            r'similar', r'suggest', r'could you recommend', r'suggestions',
            r'what to watch', r'interested in', r'worth watching'
        ]
        self.recommendation_regex = re.compile('|'.join(self.recommendation_keywords), re.IGNORECASE)


    def is_greeting(self, query):
        """
        Checks if the query is a greeting or small talk using regex patterns.

        Args:
            query (str): The user's query.

        Returns:
            bool: True if the query is a greeting or small talk, False otherwise.
        """
        return bool(self.greeting_regex.search(query))

    def contains_movie_keywords(self, query):
        """
        Checks if the query contains movie-related keywords.

        Args:
            query (str): The user's query.

        Returns:
            bool: True if the query contains movie-related keywords, False otherwise.
        """
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in self.movie_keywords)

    def is_related_to_movies(self, query, threshold=0.85):
        """
        Determines if the query is related to movies or actors using a combination
        of rule-based checks and zero-shot classification.

        Args:
            query (str): The user's query.
            threshold (float): Confidence threshold for classification.

        Returns:
            bool: True if the query is related to movies or actors, False otherwise.
        """
        # Rule-based filtering for greetings and small talk
        if self.is_greeting(query):
            logging.debug(f"Query classified as greeting/small talk: '{query}'")
            return False

        if self.contains_movie_keywords(query):
            logging.debug(f"Query contains movie keywords: '{query}'")
            return True

        labels = ["Movie-related question", "Casual conversation or greeting"]
        hypothesis_template = "This text is {}."
        result = self.classifier(
            query,
            candidate_labels=labels,
            hypothesis_template=hypothesis_template,
            multi_label=False
        )
        top_label = result['labels'][0]
        top_score = result['scores'][0]

        logging.debug(f"Zero-shot classification result: {result} - {top_score}")

        return top_label == "Movie-related question" and top_score >= threshold


    def contains_keywords(self, query, pattern):
        """
        Checks if the query matches any keywords in the provided regex pattern.

        Args:
            query (str): The user's query.
            pattern (re.Pattern): Compiled regex pattern of keywords.

        Returns:
            bool: True if a keyword match is found, False otherwise.
        """
        return bool(pattern.search(query))


    def classify_intent(self, query, threshold=0.70):
        """
        Classifies the user's intent based on keywords and zero-shot classification.

        Args:
            query (str): The user's query.
            threshold (float): Confidence threshold for classification.

        Returns:
            str: The classified intent ("Recommendation Request", "Multimedia Request", or "Other").
        """
        query = query.lower()

        result = self.classifier(
            query,
            candidate_labels=self.intent_labels,
            hypothesis_template="The user is asking for a {}.",
            multi_label=False
        )

        top_label = result['labels'][0]
        top_score = result['scores'][0]
        logging.debug(f"Zero-shot classification result: {result}")

        print(result)

        if top_score >= threshold:
            return top_label
        else:
            return "Other"