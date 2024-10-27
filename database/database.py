import os
import pandas as pd
import json
import unicodedata
import re


class DataBase:
    """Handles context data extraction for people and movies from a database with fuzzy matching support."""

    def __init__(self, db_path=None, entities_path=None):
        module_dir = os.path.dirname(os.path.abspath(__file__))

        if db_path is None:
            db_path = os.path.join(module_dir, "extended_graph_triples.pkl")
        if entities_path is None:
            entities_path = os.path.join(module_dir, "entity_db.json")

        self.db = pd.read_pickle(db_path)

        with open(entities_path, encoding="utf-8") as f:
            self.entities = json.load(f)
            self.entity_list = [subject.lower() for subject, _ in self.entities.values()]
            # Create a mapping from entity names to IDs for faster lookup
            self.name_to_id = {subject.lower(): key for key, (subject, _) in self.entities.items()}

        self.db['subject_id'] = self.db['subject_id'].astype(str).str.strip()

    @staticmethod
    def normalize_string(s):
        """Normalizes strings by removing non-ASCII characters, punctuation, and redundant spaces."""
        return ' '.join(re.sub(r'[^\w\s]', '', unicodedata.normalize('NFKD', s.lower())
                               .encode('ascii', 'ignore').decode('utf-8')).split())

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
