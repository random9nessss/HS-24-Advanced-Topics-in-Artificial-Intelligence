import random
import math
import json
import logging
from collections import defaultdict
import heapq
import pandas as pd
import datetime

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

        # Blacklisted movies
        self.blacklist = [
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
            "mystery",
            "hercules",
            "family",
            "flashback",
            "lorenzo",
            "romance",
            "rhapsody in blue",
            "forest"
        ]

        # Load the database
        self.db = pd.read_pickle(db_path)
        # Construct the graph
        self.G_nx, self.movie_release_years = construct_graph(self.db)
        # Build prefix trees and get movie values
        self.tries, self.movie_values = self.build_prefix_trees(movie_data_path, people_data_path, genre_data_path, min_title_length, min_name_length)
        # Precomputed movie dictionary for faster returns
        self.movie_details = self.build_movie_details()

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
        normalized_query = normalized_query.replace("animation", "animated")
        tokens = normalized_query.split()
        return tokens

    def build_movie_details(self):
        """
        Build a dictionary of movie details from the database.
        Returns:
            dict: A dictionary mapping movie titles to their details.
        """
        movie_details = {}
        relevant_predicates = ["director", "genre", "publication date", "imdb id", "cast member"]
        db_filtered = self.db[self.db['predicate_label'].isin(relevant_predicates)]

        for movie_id, group in db_filtered.groupby('subject_id'):
            movie_title = group['subject_label'].iloc[0]
            details = {}
            directors = set()
            genres = set()
            publication_dates = set()
            imdb_ids = set()
            cast_members = set()

            for _, row in group.iterrows():
                predicate = row['predicate_label']
                obj = row['object_label']
                if predicate == 'director':
                    directors.add(obj)

                elif predicate == 'genre':
                    genres.update([genre.strip() for genre in obj.split(',')])

                elif predicate == 'publication date':
                    date = obj.split('T')[0]
                    publication_dates.add(date)

                elif predicate == 'imdb id':
                    imdb_ids.add(obj)

                elif predicate == 'cast member':
                    cast_members.add(obj)

            details['director'] = ', '.join(directors) if directors else 'Unknown'
            details['genres'] = ', '.join(genres) if genres else 'Unknown'
            details['publication_date'] = ', '.join(publication_dates) if publication_dates else 'Unknown'

            if imdb_ids:
                imdb_id = next(iter(imdb_ids))
                details['imdb_url'] = f"imdb:{imdb_id}"
                details['has_imdb_id'] = True
            else:
                details['imdb_url'] = 'Not Available'
                details['has_imdb_id'] = False

            details['cast'] = ', '.join(cast_members) if cast_members else 'Unknown'
            details['subject_id'] = movie_id
            details['title'] = movie_title

            if movie_title in movie_details:
                movie_details[movie_title].append(details)

            else:
                movie_details[movie_title] = [details]

        # Handle duplicate movie titles -> e.g family film
        final_movie_details = {}
        for title, details_list in movie_details.items():
            def sort_key(d):
                has_imdb = d['has_imdb_id']
                pub_date = d['publication_date']

                try:
                    year = int(pub_date.split('-')[0])
                except:
                    year = 0
                return (-int(has_imdb), -year)

            sorted_details = sorted(details_list, key=sort_key)
            best_details = sorted_details[0]
            final_movie_details[title] = best_details

        return final_movie_details


    def extract_entities(self, query):
        """
        Extract entities from the query using the provided Prefix Trees.
        Prioritizes exact matches over approximate ones.
        """

        tokens = self.tokenize_query(query)
        matched_entities = defaultdict(set)
        i = 0
        while i < len(tokens):
            exact_matches = []
            approx_matches = []

            for entity_type, trie in self.tries.items():
                ematch, eend_index = trie.search_approximate(tokens, i, max_edits=0)
                if ematch:
                    exact_matches.append((entity_type, ematch, eend_index))

            if not exact_matches:
                for entity_type, trie in self.tries.items():
                    amatch, aend_index = trie.search_approximate(tokens, i, max_edits=1)
                    if amatch:
                        approx_matches.append((entity_type, amatch, aend_index))

            if exact_matches:
                entity_type, match_val, end_index = exact_matches[0]
                matched_entities[entity_type].add(match_val)
                i = end_index + 1

            elif approx_matches:
                entity_type, match_val, end_index = approx_matches[0]
                matched_entities[entity_type].add(match_val)
                i = end_index + 1
            else:
                i += 1

        # Fallback search if no entities are found (neither exact nor approximate)
        if not matched_entities:
            i = 0
            while i < len(tokens):
                fallback_matches = []
                for entity_type, trie in self.tries.items():
                    fmatch, fend_index = trie.search_approximate_merged(tokens, i, max_edits=0, max_merge_levels=2)
                    if fmatch:
                        fallback_matches.append((entity_type, fmatch, fend_index))

                if fallback_matches:
                    entity_type, match_val, end_index = fallback_matches[0]
                    matched_entities[entity_type].add(match_val)
                    i = end_index + 1
                else:
                    i += 1

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

        normalized_movie_titles = {}
        for movie_id, movie_title in movie_data.items():
            normalized_title = normalize_string(movie_title)
            title_length = len(normalized_title.replace(' ', ''))
            if title_length >= min_title_length and normalized_title not in self.blacklist:
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

            splitted_genre = normalized_genre.replace("film", "")

            if splitted_genre == "animated":
                genre_trie.insert(["animation"], genre)

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


    def rp_beta_recommendations_aggregate(self, extracted_entities, num_walks=300, walk_length_range=(2, 3),
                                          beta_range=(0, 0.05), top_n=20):
        """
        Generate movie recommendations based on random walks with dynamic edge weights and genre penalties.

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

        if len(entities) > 1:
            all_genres = []
            all_directors = []
            for entity in entities:
                if entity in self.movie_details:
                    details = self.movie_details[entity]
                    all_genres.append(set(details.get('genres', '').split(', ')))
                    all_directors.append(set(details.get('director', '').split(', ')))

            common_genres = set.intersection(*all_genres) if all_genres else set()
            common_directors = set.intersection(*all_directors) if all_directors else set()

            if common_genres or common_directors:
                logger.info("Using intersections for recommendations.")
                entities.update({item for item in common_genres if item and item != "Unknown"})
                entities.update({item for item in common_directors if item and item != "Unknown"})
                logger.info(entities)

        base_predicate_weights = {
            "director": 6,
            "genre": 7,
            "screenwriter": 4,
            "cast member": 3,
            "performer": 2,
            "publication date": 2,
            "mpaa film rating": 2,
            "production company": 3,
            "followed by": 6,
            "follows": 6
        }

        # Dynamic Edge Weight Adjustment
        dynamic_weights = base_predicate_weights.copy()

        if 'people' in entity_types and 'genres' in entity_types:
            dynamic_weights["cast member"] += 3
            dynamic_weights["genre"] += 4

        if 'genres' in entity_types and not 'people' in entity_types:
            dynamic_weights["genre"] += 10

        if 'movies' in entity_types:
            dynamic_weights["director"] += 5
            dynamic_weights["screenwriter"] += 2
            dynamic_weights["genre"] += 3
            dynamic_weights["mpaa film rating"] += 4
            dynamic_weights["production company"] += 3
            dynamic_weights["followed by"] += 8
            dynamic_weights["follows"] += 6

        if 'director' in entity_types:
            dynamic_weights["director"] += 6

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

        # If user just prompted with one genre
        if 'genres' in extracted_entities and len(extracted_entities) == 1:
            genre = next(iter(extracted_entities['genres']))

            filtered_movies = []
            for movie, details in self.movie_details.items():
                movie_genres = {g.strip().lower() for g in details.get('genres', '').split(',')}

                if genre.lower() in movie_genres:
                    score = 1.0
                    if details.get('has_imdb_id'):
                        score += 0.5

                    pub_date = details.get('publication_date', 'Unknown').split('-')[0]
                    try:
                        year = int(pub_date)
                    except:
                        year = 0
                    score += (year / 2020)

                    filtered_movies.append((movie, score))

            filtered_movies.sort(key=lambda x: x[1], reverse=True)
            recommended_movies = [(m, s) for m, s in filtered_movies if m not in self.blacklist][:top_n]

            if len(recommended_movies) < top_n:
                additional_movies = [m for m in self.movie_values if m not in [rm[0] for rm in recommended_movies]]
                random.shuffle(additional_movies)
                for m in additional_movies:
                    if len(recommended_movies) >= top_n:
                        break
                    recommended_movies.append((m, 0))

            return recommended_movies, extracted_entities

        # Recommendations if user prompted with more than just one genre
        recommendations = self.rp_beta_recommendations_aggregate(
            extracted_entities=extracted_entities,
            num_walks=300,
            walk_length_range=(1, 3),
            beta_range=(0, 0.05),
            top_n=20
        )

        entities = set()
        for entity_list in extracted_entities.values():
            entities.update(entity_list)

        recommended_movies = []
        for movie, score in recommendations:
            if movie not in entities and movie in self.movie_values and movie not in self.blacklist:
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

        return recommended_movies[:top_n], extracted_entities