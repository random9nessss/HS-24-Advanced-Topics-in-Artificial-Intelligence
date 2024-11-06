import heapq

import numpy as np
import pandas as pd


def recommend(node_label, entity_id, context, db):

    node_label = db.normalize_string(node_label)
    if node_label in db.people_names:
        movie_ids = db.people_movie_mapping[entity_id]
        context = db.fetch(movie_ids, "subject_id")
        subject_labels = context["node label"].tolist()
        return sorted(subject_labels, key=len)[:3]

    context.dropna(axis=1, inplace=True)
    columns = [col for col in context.columns if col in db.movie_recommender_db.columns]
    red_db = db.movie_recommender_db[columns]
    context = context[columns]

    # drop the identified row already in the context
    subject_id_to_remove = context["subject_id"].values[0]
    red_db = red_db[red_db["subject_id"] != subject_id_to_remove]

    red_db.dropna(thresh=len(red_db.columns) - 0, inplace=True)
    red_db.reset_index(drop=True, inplace=True)

    COLUMN_WEIGHTS = {
        'director': 0.3,
        'performer': 0.1,
        'genre': 0.3,
        'screenwriter': 0.2,
        'cast member': 0.1
    }

    def calculate_similarity(i, row):
        similarities = []
        for col in context.columns:
            if pd.isna(row[col]):
                continue

            if col in ["node label", "subject_id"]:
                continue

            set_context = set(context[col].iloc[0].split(","))
            set_row = set(row[col].split(","))
            similarity = len(set_context.intersection(set_row)) / len(set_context.union(set_row))
            similarities.append(similarity * COLUMN_WEIGHTS[col])

        return i, np.mean(similarities) if similarities else 0

    top_scores = []
    for i, row in red_db.iterrows():
        index, score = calculate_similarity(i, row)

        if len(top_scores) < 3:
            heapq.heappush(top_scores, (score, index))
        else:
            # Maintain top 3 scores
            heapq.heappushpop(top_scores, (score, index))

    # Extract indices from top 3 scores
    top_indices = [index for score, index in top_scores]
    top_rows = red_db.iloc[top_indices]

    subject_labels = top_rows['node label'].tolist()
    return subject_labels
