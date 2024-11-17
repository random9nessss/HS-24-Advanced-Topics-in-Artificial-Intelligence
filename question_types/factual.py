import random

import numpy as np
import pandas as pd
import logging
import heapq
from rapidfuzz import process, fuzz

from database.database import DataBase
from models.graph_embeddings import GraphEmbeddings
from models.ner import NERParser
from models.query_embedder import QueryEmbedderContextualized
from models.question_answering_agent import QuestionAnsweringAgent
from models.conversation_agent import ConversationAgent
from models.query_classifier import QueryClassifier

from utils.utils import (
    measure_time,
    filter_query,
    find_closest_columns,
    get_top_matches,
    fuzzy_match,
    clean_response,
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
        self.qc = QueryClassifier()
        logger.info("QueryClassifier initialized.")
        logger.info("...FactualQuestions class initialized successfully.")

    @measure_time
    def answer_query(self, query: str, last_user_query: str, last_assistant_response: str, recommender) -> str:
        logger.info(f"Query: {query}")
        normalized_query = self.db.normalize_string(query)

        ###############
        # Domain Check
        ###############
        ner_person, ner_movies = self.ner_parser.process_query(normalized_query.capitalize())

        domain_check = self.qc.is_related_to_movies(query)
        logger.info(f"Is related to movies: {domain_check}")

        if not ner_person and not ner_movies and not domain_check:
            last_assistant_response = clean_response(last_assistant_response) if last_assistant_response else ""
            small_talk = self.ca.generate_response(f"""You are a friendly and knowledgeable assistant engaged in a natural conversation. Respond to the User Query while following these guidelines:
                                        
                                                    {f'Previous response: "{last_assistant_response}".' if last_assistant_response else ''}
                                        
                                                    **User Query:** "{query}"
                                        
                                                    **Note:** The previous response may not be related to the current user query.
                                        
                                                    **Guidelines:**
                                                    1. **Context Awareness:** Use previous messages only if directly relevant to the current query.
                                                    2. **Avoid Repetition:** Don’t repeat the user's words or re-ask recent questions.{f'As such do not respond with: "{last_assistant_response}".' if last_assistant_response else ''}
                                                    3. **Follow-up Sensitivity:** Acknowledge user replies without asking similar questions.
                                        
                                                    Provide an engaging, relevant, and natural response.
                                                """)

            logger.info(f"Generated small talk response: '{small_talk}'")
            return small_talk

        fuzzy_person_match, person_full_match, person_match_length = fuzzy_match(normalized_query, self.db.people_names, self.db)
        fuzzy_movie_match, movie_full_match, movie_match_length = fuzzy_match(normalized_query, self.db.movie_names, self.db)

        fuzzy_movie_matches = []
        fuzzy_person_matches = []
        if movie_full_match and person_full_match:
            if movie_match_length > person_match_length:
                fuzzy_person_matches = []
                fuzzy_movie_matches = [fuzzy_movie_match]
            else:
                fuzzy_movie_matches = []
                fuzzy_person_matches = [fuzzy_person_match]

        elif person_full_match:
            fuzzy_movie_matches = []
            fuzzy_person_matches = [fuzzy_person_match]
        elif movie_full_match:
            fuzzy_person_matches = []
            fuzzy_movie_matches = [fuzzy_movie_match]

        elif len(fuzzy_movie_match) and len(fuzzy_person_match):
            if person_match_length > movie_match_length:
                fuzzy_movie_matches = []
            else:
                fuzzy_person_matches = []

        fuzzy_movie_context = self.db.fetch(fuzzy_movie_matches, "subject_id")
        fuzzy_person_context = self.db.fetch(fuzzy_person_matches, "subject_id")

        context = pd.concat([fuzzy_movie_context, fuzzy_person_context])

        if context.empty:
            logger.warning("No context data found for the given query.")

            # Fallback Strategy
            last_assistant_response = clean_response(last_assistant_response) if last_assistant_response else ""
            small_talk = self.ca.generate_response(f"""You are a friendly and knowledgeable assistant engaged in a natural conversation. Respond to the User Query while following these guidelines:

                                                    {f'Previous response: "{last_assistant_response}".' if last_assistant_response else ''}

                                                    **User Query:** "{query}"

                                                    **Note:** The previous response may not be related to the current user query.

                                                    **Guidelines:**
                                                    1. **Context Awareness:** Use previous messages only if directly relevant to the current query.
                                                    2. **Avoid Repetition:** Don’t repeat the user's words or re-ask recent questions.{f'As such do not respond with: "{last_assistant_response}".' if last_assistant_response else ''}
                                                    3. **Follow-up Sensitivity:** Acknowledge user replies without asking similar questions.

                                                    Provide an engaging, relevant, and natural response.
                                                """)

            logger.info(f"Generated small talk response: '{small_talk}'")
            return small_talk


        context = get_top_matches(context, normalized_query, top_n=1)

        node_label = ""
        if "node label" in context.columns and not context["node label"].isna().values[0]:
            node_label = context["node label"].values[0]
        else:
            pass

        entity_id = ""
        if "subject_id" in context.columns and not context["subject_id"].isna().values[0]:
            entity_id = context["subject_id"].values[0]
        else:
            pass

        # ************MULTIMEDIA************

        #intent = self.qc.classify_intent(query)

        keywords = [
            "picture",
            "image",
            "poster",
            "photo",
            "show",
            "look"
        ]
        if any([word for word in keywords if word in query]) and "imdb id" in context.columns and not context["imdb id"].isna().values[0]:
            logger.info("Detected multimedia query.")
            imdb_id = context["imdb id"].values[0]
            image = self.db.get_image(imdb_id, is_movie=True if fuzzy_movie_matches else False)
            if image and node_label:
                return f"Here is a Picture of {node_label}\n image:{image}"
            return f"Sorry, I don't find any image for {node_label}."
        # ************MULTIMEDIA************

        # ************RECOMMENDATION************
        keywords = [
            "recommend",
            "propose",
            "advise",
            "offer",
            "like",
            "should",
            "must",
            "similar",
            "suggest"
        ]

        if any([word for word in keywords if word in query]):
            logger.info("Detected recommendation query.")
            recommended_movies, identified_entities = recommender.recommend_movies(query)
            logger.info(recommended_movies.replace("\n", ""))

            formatted_recommendation = (
                f"Based on your interest in: {identified_entities}\n\n"
                f"I recommend the following movies:\n\n"
                f"{recommended_movies}\n\n"
                "Enjoy your movie time!"
            )

            return formatted_recommendation

        # ************RECOMMENDATION************


        # Remove unused columns
        elements_to_remove = ["image", "color", "sport"]
        context = context.drop(columns=elements_to_remove, errors='ignore')
        #logger.debug(f"Context after removing unused columns: {context.columns}")

        # Initial context for embeddings where original column names are required
        initial_context = context.copy()

        # Rename columns
        columns_to_rename = {
            "cast member": "movie cast",
            "notable work": "acted in"
        }
        context = context.rename(columns={k: v for k, v in columns_to_rename.items() if k in context.columns})
        #logger.debug(f"Context after renaming columns: {context.columns}")

        columns_to_duplicate = [("acted in", "played in"),
                                ("acted in", "appeared in"),
                                ("movie cast", "actors"),
                                ("movie cast", "players")]

        for col_to_duplicate, col in columns_to_duplicate:
            if col_to_duplicate in context.columns:
                context[col] = context[col_to_duplicate]
                #logger.debug(f"Duplicated column '{col_to_duplicate}' to '{col}'")

        context.dropna(axis=1, inplace=True)
        #logger.debug(f"Context after dropping NaNs: {context.columns}")

        query_filtered = filter_query(query, node_label)
        #logger.debug(f"Filtered query: '{query_filtered}'")

        column_embeddings = {col: self.qe.embed_phrase(col) for col in context.columns}
        #logger.debug("Column embeddings computed.")
        query_embeddings = [self.qe.embed_phrase(word) for word in query_filtered.split()]
        #logger.debug("Query embeddings computed.")
        top_columns_embeddings = find_closest_columns(query_embeddings, column_embeddings)
        #logger.debug(f"Top columns from embeddings: {top_columns_embeddings}")

        # Always keep these columns
        col_always_keep = ["node label"]

        combined_columns = set(top_columns_embeddings + col_always_keep)
        top_columns = [col for col in combined_columns if col in context.columns]
        #logger.debug(f"Final selected columns: {top_columns}")
        filtered_context_df = context[top_columns]

        answer = self.qa.query(query, filtered_context_df)
        #logger.debug(f"Answer from QA model: '{answer}'")
        formatted_answer = self.ca.generate_response(f"""You are a knowledgeable assistant specializing in movies. Your goal is to provide a clear and accurate response based on the given answer, ensuring it sounds natural and relevant to the user.
                                                
                                                     Question: "{query}"
                                                     Given Answer: "{answer}"
                                                
                                                     Instructions:
                                                     1. **Validate the Given Answer**: Carefully read the provided answer. If it appears accurate and relevant to the question, rephrase it in a conversational and polished way for the user.
                                                     2. **Override Only if Necessary**: Only override the answer if it is completely nonsensical, irrelevant to movies, or obviously incorrect. Use your expertise in movies to provide a more accurate response in these cases.
                                                     3. **Keep the Focus on Movies**: Remember that you are a movie bot, so your responses should naturally incorporate movie-related knowledge when necessary.
                                                
                                                     Provide a final response that sounds natural and trustworthy.
                                                 """)

        logger.info(f"Final answer: '{formatted_answer}'")

        embedding_answer = self.ge.answer_query_embedding(initial_context, top_columns)

        return f"Graph:\n{formatted_answer}\n\nEmbeddings:\n{embedding_answer}"