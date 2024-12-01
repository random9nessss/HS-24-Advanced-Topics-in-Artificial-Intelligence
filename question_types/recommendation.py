# recommender.py

import random
import json
import logging
from collections import defaultdict
import heapq
import pandas as pd

from PrefixTree.PrefixTree import PrefixTree
from Graph.graph_constructor import construct_graph

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STOPWORDS = {"a", "an", "the", "of", "on", "and", "in", "at", "for", "to", "is", "it"}

def normalize_string(s):
    """
    Normalize the input string by converting it to lowercase and removing special characters.
    """
    import re
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', '', s)
    return ' '.join(word for word in s.split() if word not in STOPWORDS)


class Recommender:
    """
    Recommender class that precomputes the graph and prefix trees and provides a method to recommend movies.
    """

    def __init__(self, db_path, movie_data_path, people_data_path, genre_data_path, min_title_length=4, min_name_length=5):
        """
        Initialize the Recommender by constructing the graph and building prefix trees.

        Args:
            db_path (str): Path to the database CSV file.
            movie_data_path (str): Path to the movie data JSON file.
            people_data_path (str): Path to the people data JSON file.
            genre_data_path (str): Path to the genre data JSON file.
            min_title_length (int): Minimum length of movie titles to include.
            min_name_length (int): Minimum length of people names to include.
        """
        logger.info("Initializing Recommender class...")
        # Load the database
        self.db = pd.read_pickle(db_path)
        # Construct the graph
        self.G_nx = construct_graph(self.db)
        # Build prefix trees and get movie values
        self.tries, self.movie_values = self.build_prefix_trees(movie_data_path, people_data_path, genre_data_path, min_title_length, min_name_length)

        logger.info("...Recommender class initialized successfully")

    @staticmethod
    def tokenize_query(query):
        """
        Tokenize and normalize the query string.

        Args:
            query (str): The query string to tokenize.

        Returns:
            list of str: A list of normalized tokens from the query.
        """
        normalized_query = normalize_string(query)
        tokens = normalized_query.split()
        return tokens

    def extract_entities(self, query):
        """
        Extract entities from the query using the provided Prefix Trees.

        Args:
            query (str): The query string from which to extract entities.

        Returns:
            dict: A dictionary with entity types as keys and sets of extracted entities as values.
        """
        tokens = self.tokenize_query(query)
        matched_entities = defaultdict(set)
        i = 0
        while i < len(tokens):
            match_found = False
            for entity_type, trie in self.tries.items():
                match, end_index = trie.search_approximate(tokens, i, max_edits=1)
                if match:
                    matched_entities[entity_type].add(match)
                    i = end_index + 1  # Move past the matched entity
                    match_found = True
                    break
            if not match_found:
                i += 1  # Move to the next token if no match

        return matched_entities

    def build_prefix_trees(self, movie_data_path, people_data_path, genre_data_path, min_title_length=4, min_name_length=4, min_genre_length=4):
        """
        Build prefix trees for movies, people, and genres.

        Args:
            movie_data_path (str): Path to the movie data JSON file.
            people_data_path (str): Path to the people data JSON file.
            genre_data_path (str): Path to the genre data JSON file.
            min_title_length (int): Minimum length of movie titles to include.
            min_name_length (int): Minimum length of people names to include.

        Returns:
            dict: A dictionary containing the prefix trees for movies, people, and genres.
            set: A set of movie titles.
        """
        with open(movie_data_path) as f:
            movie_data = json.load(f)

        blacklist = [
            "performance",
            "watch",
            "love",
            "them",
            "play",
            "player",
            "good",
            "mother",
            "nice",
            "film",
            "method",
            "linked",
            "string",
            "message",
            "after",
            "westen",
            "what man",
            "ryan",
            "others",
            "loved",
            "appearead",
            "actors",
            "star",
            "mystery"
        ]

        normalized_movie_titles = {}
        for movie_id, movie_title in movie_data.items():
            normalized_title = normalize_string(movie_title)
            title_length = len(normalized_title.replace(' ', ''))
            if title_length >= min_title_length and normalized_title not in blacklist:
                normalized_movie_titles[normalized_title] = movie_title

        with open(people_data_path) as f:
            people_data = json.load(f)

        normalized_people_names = {}
        for person_id, person_name in people_data.items():
            normalized_name = normalize_string(person_name)
            name_length = len(normalized_name.replace(' ', ''))
            if name_length >= min_name_length:
                normalized_people_names[normalized_name] = person_name

        movie_trie = PrefixTree()
        for normalized_title, original_title in normalized_movie_titles.items():
            title_tokens = normalized_title.split()
            movie_trie.insert(title_tokens, original_title)

        people_trie = PrefixTree()
        for normalized_name, original_name in normalized_people_names.items():
            name_tokens = normalized_name.split()
            people_trie.insert(name_tokens, original_name)

        with open(genre_data_path) as f:
            genre_data = json.load(f)

        genre_trie = PrefixTree()
        for genre in genre_data:
            normalized_genre = normalize_string(genre)
            genre_tokens = normalized_genre.replace("film", "").strip().split()

            if len(normalized_genre) >= min_genre_length:
                genre_trie.insert(genre_tokens, genre)

        tries = {
            'movies': movie_trie,
            'people': people_trie,
            'genres': genre_trie
        }

        movie_values = set(movie_data.values())

        return tries, movie_values

    def rp_beta_recommendations_aggregate(self, extracted_entities, num_walks=300, walk_length_range=(2, 3), beta_range=(0, 0.05), top_n=20):
        """
        Generate movie recommendations based on random walks with dynamic edge weights.

        Args:
            extracted_entities (dict): Dictionary of entity types to sets of entities extracted from the query.
            num_walks (int): Number of random walks to perform.
            walk_length_range (tuple): Range of walk lengths.
            beta_range (tuple): Range of beta values for attenuation.
            top_n (int): Number of top recommendations to return.

        Returns:
            list of tuples: List of recommended movies with their scores.
        """
        scores = defaultdict(float)
        entities = set()
        for entity_list in extracted_entities.values():
            entities.update(entity_list)

        entity_types = set(extracted_entities.keys())

        base_predicate_weights = {
            "director": 5,
            "genre": 5,
            "screenwriter": 3,
            "cast member": 1,
            "performer": 2,
            "publication date": 2,
            "mpaa film rating": 2
        }

        # Dynamic Edge Weight Adjustment
        dynamic_weights = base_predicate_weights.copy()

        if 'people' in entity_types and 'genres' in entity_types:
            dynamic_weights["cast member"] += 4
            dynamic_weights["genre"] += 3

        if 'genres' in entity_types and not 'people' in entity_types:
            dynamic_weights["genre"] += 7

        if 'movies' in entity_types:
            dynamic_weights["director"] += 2
            dynamic_weights["screenwriter"] += 1
            dynamic_weights["genre"] += 2

        if 'director' in entity_types:
            dynamic_weights["director"] += 4

        max_weight = max(dynamic_weights.values())
        for key in dynamic_weights:
            dynamic_weights[key] = dynamic_weights[key] / max_weight

        for u, v, data in self.G_nx.edges(data=True):
            predicate_label = data.get('predicate_label')
            base_weight = data.get('weight', 1)
            adjusted_weight = base_weight * dynamic_weights.get(predicate_label, 1)
            self.G_nx[u][v]['adjusted_weight'] = adjusted_weight

        for entity in entities:
            if entity not in self.G_nx:
                continue
            for _ in range(num_walks):
                walk_length = random.randint(*walk_length_range)
                beta = random.uniform(*beta_range)
                path = [entity]
                current_node = entity
                for _ in range(walk_length):
                    neighbors = list(self.G_nx.neighbors(current_node))
                    if not neighbors:
                        break

                    weights = []
                    for neighbor in neighbors:
                        edge_data = self.G_nx.get_edge_data(current_node, neighbor)
                        weights.append(edge_data.get('adjusted_weight', 1))

                    total_weight = sum(weights)
                    probabilities = [w / total_weight for w in weights]

                    next_node = random.choices(neighbors, weights=probabilities, k=1)[0]
                    path.append(next_node)
                    current_node = next_node
                for node in path:
                    if node in self.movie_values and node not in entities:
                        scores[node] += 1 * (1 - beta)

        max_score = max(scores.values(), default=1)
        for movie in scores:
            scores[movie] /= max_score

        top_recommendations = heapq.nlargest(top_n, scores.items(), key=lambda x: x[1])

        return top_recommendations

    def recommend_movies(self, query, top_n=5):
        """
        Generate movie recommendations based on the user query.

        Args:
            query (str): The user query.
            top_n (int): Number of recommendations to return.

        Returns:
            list of tuples: Recommended movies with their scores.
            dict: Extracted entities with their types.
        """
        extracted_entities = self.extract_entities(query)
        logger.info(f"Extracted Entities: {extracted_entities}")

        recommendations = self.rp_beta_recommendations_aggregate(
            extracted_entities=extracted_entities,
            num_walks=500,
            walk_length_range=(1, 3),
            beta_range=(0, 0.05),
            top_n=20
        )

        entities = set()
        for entity_list in extracted_entities.values():
            entities.update(entity_list)

        recommended_movies = []
        for movie, score in recommendations:
            if movie not in entities and movie in self.movie_values:
                recommended_movies.append((movie, score))
            if len(recommended_movies) >= top_n:
                break

        # Fallback: Random Recommendations
        if len(recommended_movies) < top_n:
            additional_movies = [m for m in self.movie_values if m not in entities and m not in [rm[0] for rm in recommended_movies]]
            random.shuffle(additional_movies)
            for m in additional_movies:
                recommended_movies.append((m, 0))
                if len(recommended_movies) >= top_n:
                    break

        return recommended_movies, extracted_entities