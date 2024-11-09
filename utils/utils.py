import torch
import numpy as np
import pandas as pd
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from rapidfuzz import process, fuzz
import time
from functools import wraps
import logging

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Download stopwords
nltk.download('stopwords', quiet=True)

stop_words_to_keep = [
    "what", "when", "where", "which", "while", "who", "whom", "why",
    "with", "how", "before", "after", "same"
]
stop_words = set([s for s in stopwords.words('english') if s not in stop_words_to_keep])

# Define colors for logging
class BColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

# Custom formatter to color log levels
class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.COLORS = {
            'DEBUG': BColors.OKBLUE,
            'INFO': '',  # No color for INFO messages
            'WARNING': BColors.WARNING,
            'ERROR': BColors.FAIL,
            'CRITICAL': BColors.BOLD + BColors.FAIL,
        }

    def format(self, record):
        levelname = record.levelname.strip(BColors.ENDC)
        level_color = self.COLORS.get(levelname, '')
        record.levelname = level_color + record.levelname + BColors.ENDC
        return super().format(record)

# Set up logger
logger = logging.getLogger('factual_questions')
logger.setLevel(logging.INFO)

# Create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter
formatter = ColoredFormatter('%(asctime)s | %(levelname)s | %(funcName)s | %(message)s')

# Add formatter to handler
ch.setFormatter(formatter)

# Add handler to logger if not already added
if not logger.hasHandlers():
    logger.addHandler(ch)

def get_device():
    """Determines the available hardware device (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the function
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        # Log the execution time under INFO level, with time in green color
        elapsed_time_str = f"{BColors.OKGREEN}{elapsed_time:.4f} seconds{BColors.ENDC}"
        logger.info(f"Execution time for {func.__name__}: {elapsed_time_str}")
        return result
    return wrapper

def cosine_sim(vec1, vec2):
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    return np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)

def rescale_probabilities(similarities):
    """
    Rescales the similarity scores so that they sum to 1, turning them into a probability distribution.

    Args:
        similarities (List[float]): List of similarity scores.

    Returns:
        List[float]: Rescaled probabilities.
    """
    similarity_sum = sum(similarities)
    if similarity_sum == 0:
        return [0] * len(similarities)  # Avoid division by zero

    return [sim / similarity_sum for sim in similarities]

def find_closest_columns(query_embeddings, column_embeddings, high_threshold=0.4, top_n=10, rescaled_threshold=0.11):
    """
    Returns columns based on cosine similarity with a two-tiered strategy and rescaled probabilities.
    - If a column has similarity above 'high_threshold', return that column immediately.
    - Otherwise, return all columns with a rescaled probability greater than 'rescaled_threshold'.

    Args:
        query_embeddings (List[np.ndarray]): Embeddings for query words.
        column_embeddings (Dict[str, np.ndarray]): Precomputed embeddings for columns.
        high_threshold (float): Confidence threshold to return immediately (default: 0.4).
        top_n (int): Number of top columns to consider for rescaling (default: 10).
        rescaled_threshold (float): Minimum rescaled probability threshold (default: 0.11).

    Returns:
        List[str]: The selected column names.
    """
    column_similarities = {}

    for col, col_vec in column_embeddings.items():
        similarities = [cosine_sim(col_vec, q_vec) for q_vec in query_embeddings if np.linalg.norm(q_vec) > 0]
        column_similarities[col] = np.mean(similarities) if similarities else -1

    sorted_columns = sorted(column_similarities.items(), key=lambda item: item[1], reverse=True)
    top_columns = sorted_columns[:top_n]

    column_names, similarities = zip(*top_columns)

    rescaled_probs = rescale_probabilities(similarities)

    selected_columns = []

    for col, sim in zip(column_names, similarities):
        if sim >= high_threshold:
            logger.info(f"High confidence match found: {col} with similarity {sim:.4f}")
            return [col]

    for col, rescaled_prob in zip(column_names, rescaled_probs):
        if rescaled_prob >= rescaled_threshold:
            logger.info(f"Column {col} has rescaled similarity {rescaled_prob:.4f}")
            selected_columns.append(col)

    return selected_columns

def filter_query(query, node_label):
    if not query:
        return ''

    relevant = []
    node_label_cleaned = node_label.lower().replace(" ", "")
    for word in query.replace(". ", " ").lower().split():
        cleaned_word = re.sub(r'[^A-Za-z]', '', word)
        if cleaned_word in stop_words or cleaned_word in node_label_cleaned or not cleaned_word:
            continue
        relevant.append(cleaned_word)

    return " ".join(relevant)


def fuzzy_match(query_str, comparison_list, db):
    if not comparison_list or not query_str:
        return []

    name_to_id = {v: k for k, v in db.entities.items()}

    longest_full_match = ""
    longest_full_length = 0
    longest_prefix_match = ""
    longest_prefix_length = 0

    query_str = db.normalize_string(query_str)

    # Loop through the comparison list to find both longest full match and longest prefix match
    for subject in comparison_list:
        if "porn" in subject:
            continue

        if subject in query_str:
            if len(subject) > longest_full_length:
                longest_full_match = subject
                longest_full_length = len(subject)

        # Check for longest prefix match
        for i in range(len(subject), 0, -1):
            if subject[:i] == query_str[:i] and len(subject) > len(query_str):
                if i > longest_prefix_length:
                    longest_prefix_match = subject
                    longest_prefix_length = i
                break

    if longest_full_length >= 4:
        logger.info(f"Found FULL match: {longest_full_match}")
        return [name_to_id[longest_full_match]]
    elif longest_prefix_length >= 9:
        logger.info(f"Found PREFIX match: {longest_prefix_match}")
        return [name_to_id[longest_prefix_match]]
    return []


def get_top_matches(df, query_str, top_n=1):
    """
    Given a DataFrame with a 'node label' column, returns the top N rows where 'node label' best matches the query string.
    """
    if df.empty or 'node label' not in df.columns:
        return pd.DataFrame()

    # Use fuzzy matching to find the best matching node labels
    node_labels = df['node label'].tolist()
    matches = process.extract(query_str, node_labels, scorer=fuzz.partial_ratio, limit=top_n)

    # Get the matched labels
    matched_labels = [match[0] for match in matches]

    # Return the rows where 'node label' is in matched_labels
    return df[df['node label'].isin(matched_labels)]


def clean_response(response: str) -> str:
    response = re.sub(r'^Graph:\s*', '', response, flags=re.MULTILINE)
    response = re.sub(r'Embeddings:.*', '', response, flags=re.DOTALL).strip()
    return response