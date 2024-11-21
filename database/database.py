import random

import pandas as pd
import json
import re
import unicodedata
from pathlib import Path

class DataBase:
    """Handles context data extraction for people and movies from a database with fuzzy matching support."""

    def __init__(self):
        # Use the current file's directory to locate the "dataset" folder
        module_dir = Path(__file__).resolve().parent.parent
        dataset_dir = module_dir / "dataset"

        with open(dataset_dir / 'images.json') as f:
            self.images_data = json.load(f)

        with open(dataset_dir / 'image_index_cleaned.json') as f:
            self.image_lookup = json.load(f)

        # Load the main database
        self.db = pd.read_pickle(dataset_dir / "extended_graph_triples.pkl")
        self.db['subject_id'] = self.db['subject_id'].astype(str).str.strip()
        self.db['predicate_label'] = self.db['predicate_label'].astype(str).str.strip()
        self.db['object_label'] = self.db['object_label'].astype(str).str.strip()

        # Create a pivot table from the database
        self.db_pivot = self.db.pivot_table(
            index='subject_id',
            columns='predicate_label',
            values='object_label',
            aggfunc=lambda x: ' | '.join(x.astype(str))
        )

        # Load JSON data for movies and people
        with open(dataset_dir / 'movie_db.json') as f:
            self.movie_data = json.load(f)
            self.movie_ids = set(self.movie_data.keys())
        self.movie_db = pd.DataFrame(list(self.movie_data.items()), columns=["entity_id", "entity_label"])

        with open(dataset_dir / 'people_db.json') as f:
            self.people_data = json.load(f)
            self.people_ids = set(self.people_data.keys())
        self.people_db = pd.DataFrame(list(self.people_data.items()), columns=["entity_id", "entity_label"])

        # Merge entities, movie names, and people names
        self.entities = {**self.movie_data, **self.people_data}
        self.movie_names = self.movie_db["entity_label"].tolist()
        self.people_names = self.people_db["entity_label"].tolist()
        self.entity_list = self.movie_names + self.people_names

        # Initialize mappings and recommender database
        self.people_movie_mapping = {}
        self.movie_people_mapping = {}
        self.map_people_movies(dataset_dir)
        self.movie_recommender_db = self.filter_relevant_movies()

        # Load corrected crowd data
        with open(dataset_dir / 'crowd_source.json') as f:
            self.crowd_data = json.load(f)


    def get_image(self, imdb_id, is_movie=True):
        if imdb_id not in self.image_lookup:
            return ""

        if is_movie:
            priorities = [
                "poster",
                "publicity",
                "still_frame",
                "product",
                "behind_the_scenes",
                "event",
                "production_art",
                "unknown",
                "all"
            ]
        else:
            priorities = [
                "publicity",
                "still_frame",
                "poster",
                "behind_the_scenes",
                "event",
                "product",
                "production_art",
                "user_avatar",
                "unknown",
                "all"
            ]

        images = []
        for img_type in priorities:
            if img_type in self.image_lookup[imdb_id]:
                images = self.image_lookup[imdb_id][img_type]
                break

        return random.choice(images) if images else ""


    def map_people_movies(self, dataset_dir):
        # Load triples only IDs and clean them
        id_triples_path = dataset_dir / "df_new_triples_only_ids.pkl"
        id_triples = pd.read_pickle(id_triples_path)
        id_triples['subject_id'] = id_triples['subject_id'].astype(str).str.strip()
        id_triples['object_id'] = id_triples['object_id'].astype(str).str.strip()

        # Map relationships
        for _, row in id_triples.iterrows():
            subject_id = row['subject_id']
            object_id = row['object_id']

            if subject_id in self.people_ids and object_id in self.movie_ids:
                self.people_movie_mapping.setdefault(subject_id, []).append(object_id)
                self.movie_people_mapping.setdefault(object_id, []).append(subject_id)

        # Remove duplicates in mappings
        self.people_movie_mapping = {person: list(set(movies)) for person, movies in self.people_movie_mapping.items()}
        self.movie_people_mapping = {movie: list(set(people)) for movie, people in self.movie_people_mapping.items()}

    @staticmethod
    def normalize_string(s):
        """Normalizes strings by removing non-ASCII characters, punctuation, and redundant spaces."""
        normalized = unicodedata.normalize('NFKD', s.lower())
        cleaned = re.sub(r'[^\w\s]', '', normalized.replace("'", "").replace('"', ''))
        return ' '.join(cleaned.split())

    def fetch(self, entity_list, search_column):
        """Fetches relevant rows from the database where `search_column` matches values in `entity_list`."""
        relevant = self.db[self.db[search_column].isin(entity_list)].dropna(axis=1)

        if relevant.empty:
            return pd.DataFrame()

        return relevant.pivot_table(
            index='subject_id',
            columns='predicate_label',
            values='object_label',
            aggfunc=lambda x: ' | '.join(x.astype(str))
        ).reset_index()

    def filter_relevant_movies(self):
        clean_db = self.db[self.db["subject_id"].isin(self.movie_data.keys())]
        relevant_cols = [
            # "author",  # only 99 movies have an author
            "cast member",
            "director",
            "performer",
            "genre",
            # "narrative motif",  # only 43 movies have a narrative motif
            "screenwriter",
            "subject_id",
            "node label"  # Required for processing
        ]

        pv_db = clean_db.pivot_table(
            index='subject_id',
            columns='predicate_label',
            values='object_label',
            aggfunc=lambda x: ' | '.join(x.astype(str))
        )

        pv_db = pv_db[[col for col in relevant_cols if col in pv_db.columns]]

        pv_db = pv_db.drop_duplicates().reset_index()
        pv_db.to_pickle("../development/exports/df_recommender.pkl")

        return pv_db

if __name__ == "__main__":
    db = DataBase()