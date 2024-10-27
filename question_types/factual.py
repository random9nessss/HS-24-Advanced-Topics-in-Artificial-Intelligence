import pandas as pd
import logging

from database.database import DataBase
from models.graph_embeddings import GraphEmbeddings
from models.ner import NERParser
from models.query_embedder import QueryEmbedderContextualized
from models.question_answering_agent import QuestionAnsweringAgent
from models.conversation_agent import ConversationAgent
from utils.utils import (
    measure_time,
    filter_query,
    find_closest_columns,
    get_top_matches,
    fuzzy_match,
    logger
)

class FactualQuestions:
    def __init__(self):
        logger.info("Initializing FactualQuestions class...")
        self.db = DataBase()
        logger.info("Database initialized.")
        self.ner_parser = NERParser(lowercase=False)
        logger.info("NERParser initialized.")
        self.qe = QueryEmbedderContextualized()
        logger.info("QueryEmbedderContextualized initialized.")
        self.qa = QuestionAnsweringAgent()
        logger.info("QuestionAnsweringAgent initialized.")
        self.ca = ConversationAgent(model_name="google/flan-t5-xl")
        logger.info("ConversationAgent initialized.")
        self.ge = GraphEmbeddings(graph=self.db.db)
        logger.info("GraphEmbeddings initialized.")
        logger.info("FactualQuestions class initialized successfully.")

    @measure_time
    def answer_query(self, query):
        logger.info(f"Received query: '{query}'")
        normalized_query = self.db.normalize_string(query)
        logger.debug(f"Normalized query: '{normalized_query}'")

        entity_matches = fuzzy_match(normalized_query, self.db.entity_list, self.db, threshold=30)
        logger.debug(f"Entity matches: {entity_matches}")

        # NER Model and NER Matching
        ner_person, ner_movies = self.ner_parser.process_query(query)
        logger.debug(f"NER detected persons: {ner_person}, movies: {ner_movies}")

        if ner_movies:
            ner_movie_entities = fuzzy_match(" ".join(ner_movies), self.db.entity_list, self.db, threshold=75)
            logger.debug(f"NER movie entities: {ner_movie_entities}")
            subjects_ner_movies = self.db.fetch(ner_movie_entities, "subject_id")
            context_ner_movies = get_top_matches(subjects_ner_movies, normalized_query, top_n=1)
        else:
            context_ner_movies = pd.DataFrame()

        if ner_person:
            ner_person_entities = fuzzy_match(" ".join(ner_person), self.db.entity_list, self.db, threshold=75)
            logger.debug(f"NER person entities: {ner_person_entities}")
            subjects_ner_person = self.db.fetch(ner_person_entities, "subject_id")
            context_ner_person = get_top_matches(subjects_ner_person, normalized_query, top_n=1)
        else:
            context_ner_person = pd.DataFrame()

        is_domain_specific = bool(ner_person or ner_movies)
        logger.debug(f"Is domain specific: {is_domain_specific}")

        if not is_domain_specific:
            logger.info("Query is not domain-specific. Generating small talk response.")
            small_talk = self.ca.generate_response(query)
            logger.info(f"Generated small talk response: '{small_talk}'")
            return small_talk

        # Fuzzy Matching
        subjects = self.db.fetch(entity_matches, "subject_id")
        logger.debug(f"Fetched subjects: {subjects}")
        context = get_top_matches(subjects, normalized_query, top_n=1)
        logger.debug(f"Context after get_top_matches: {context}")

        ner_context = pd.concat([context_ner_movies, context_ner_person])
        logger.debug(f"NER context: {ner_context}")

        if not ner_context.empty:
            context = ner_context
            logger.debug("Using NER context.")

        node_label = context["node label"].values[0] if "node label" in context.columns and not context.empty else ""
        logger.debug(f"Node label: '{node_label}'")

        if context.empty:
            logger.warning("No context data found for the given query.")
            # Fallback Strategy
            small_talk = self.ca.generate_response(query)
            logger.info(f"Generated small talk response: '{small_talk}'")
            return small_talk

        # Remove unused columns
        elements_to_remove = ["image", "color", "sport"]
        context = context.drop(columns=elements_to_remove, errors='ignore')
        logger.debug(f"Context after removing unused columns: {context.columns}")

        # Initial context for embeddings where original column names are required
        initial_context = context.copy()

        # Rename columns
        columns_to_rename = {
            "cast member": "movie cast",
            "notable work": "acted in"
        }
        context = context.rename(columns={k: v for k, v in columns_to_rename.items() if k in context.columns})
        logger.debug(f"Context after renaming columns: {context.columns}")

        columns_to_duplicate = [("acted in", "played in"),
                                ("acted in", "appeared in"),
                                ("movie cast", "actors"),
                                ("movie cast", "players")]

        for col_to_duplicate, col in columns_to_duplicate:
            if col_to_duplicate in context.columns:
                context[col] = context[col_to_duplicate]
                logger.debug(f"Duplicated column '{col_to_duplicate}' to '{col}'")

        context.dropna(axis=1, inplace=True)
        logger.debug(f"Context after dropping NaNs: {context.columns}")

        query_filtered = filter_query(query, node_label)
        logger.debug(f"Filtered query: '{query_filtered}'")

        column_embeddings = {col: self.qe.embed_phrase(col) for col in context.columns}
        logger.debug("Column embeddings computed.")
        query_embeddings = [self.qe.embed_phrase(word) for word in query_filtered.split()]
        logger.debug("Query embeddings computed.")
        top_columns_embeddings = find_closest_columns(query_embeddings, column_embeddings)
        logger.debug(f"Top columns from embeddings: {top_columns_embeddings}")

        # Fuzzy matching on columns
        top_columns_fuzzy = []
        logger.debug(f"Top columns from fuzzy matching: {top_columns_fuzzy}")

        # Always keep these columns
        col_always_keep = ["node label"]

        combined_columns = set(top_columns_fuzzy + top_columns_embeddings + col_always_keep)
        top_columns = [col for col in combined_columns if col in context.columns]
        logger.debug(f"Final selected columns: {top_columns}")
        filtered_context_df = context[top_columns]

        answer = self.qa.query(query, filtered_context_df)
        logger.debug(f"Answer from QA model: '{answer}'")
        formatted_answer = self.ca.generate_response(f""""Format the answer to the question into a sentence.
                                                    If you think the answer is completely off, overrule with your own knowledge.
                                                    Question: {query}\nAnswer: {answer}""")

        logger.info(f"Final answer: '{formatted_answer}'")

        embedding_answer = self.ge.answer_query_embedding(initial_context, top_columns)

        return f"Graph:\n{formatted_answer}\n\nEmbeddings:\n{embedding_answer}"