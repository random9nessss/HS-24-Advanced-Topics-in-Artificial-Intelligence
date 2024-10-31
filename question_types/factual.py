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
        logger.info("FactualQuestions class initialized successfully.")

    @measure_time
    def answer_query(self, query: str, last_user_query: str, last_assistant_response: str) -> str:
        normalized_query = self.db.normalize_string(query)
        logger.info("Normalized query for NER processing.")

        ner_person, ner_movies = self.ner_parser.process_query(normalized_query)
        logger.info("Processed query using NER parser.")
        logger.info(f"NER_Person: {ner_person}")
        logger.info(f"NER_Movies: {ner_movies}")

        # Process people IDs
        ner_people_ids = []
        for ner_p in ner_person:
            ner_p = self.db.normalize_string(ner_p)
            logger.debug(f"Normalized NER person: {ner_p}")
            if ner_p in self.db.people_names:
                ner_people_ids = [key for key, value in self.db.people_data.items() if value == ner_p]
                logger.info(f"Direct match found for person: {ner_p}")
            if not ner_people_ids:
                ner_people_ids = [key for key, value in self.db.people_data.items() if ner_p in value]
                logger.debug(f"Partial match search in people data for: {ner_p}")
            if not ner_people_ids:
                ner_people_ids = fuzzy_match(" ".join(ner_person), self.db.people_names, threshold=75)
                logger.info("Using fuzzy match for person IDs.")

        # Process movie IDs
        ner_movie_ids = []
        for ner_m in ner_movies:
            ner_m = self.db.normalize_string(ner_m)
            logger.debug(f"Normalized NER movie: {ner_m}")
            if ner_m in self.db.movie_names:
                ner_movie_ids = [key for key, value in self.db.movie_data.items() if value == ner_m]
                logger.info(f"Direct match found for movie: {ner_m}")
            if not ner_movie_ids:
                ner_movie_ids = [key for key, value in self.db.movie_data.items() if ner_m in value]
                logger.debug(f"Partial match search in movie data for: {ner_m}")
            if not ner_movie_ids:
                ner_movie_ids = fuzzy_match(" ".join(ner_movies), self.db.movie_names, threshold=75)
                logger.info("Using fuzzy match for movie IDs.")

        # Combine IDs and fetch context
        ner_ids = ner_movie_ids + ner_people_ids
        logger.debug(f"Combined NER IDs: {ner_ids}")

        context = self.db.fetch(ner_ids, "subject_id")
        if context.empty:
            logger.warning("NER failed; proceeding with fuzzy matching.")

            fuzzy_person_matches = fuzzy_match(normalized_query, self.db.people_names, self.db, threshold=30)
            fuzzy_movie_matches = fuzzy_match(normalized_query, self.db.movie_names, self.db, threshold=30)

            logger.info(f"Fuzzy person matches: {fuzzy_person_matches}")
            logger.info(f"Fuzzy movie matches: {fuzzy_movie_matches}")

            fuzzy_movie_context = self.db.fetch(fuzzy_movie_matches, "subject_id")
            fuzzy_person_context = self.db.fetch(fuzzy_person_matches, "subject_id")

            context = pd.concat([fuzzy_movie_context, fuzzy_person_context])
            logger.info("Fuzzy matching completed and context fetched.")
            logger.info(f"Final context after fuzzy matching: {context}")

        if context.empty:
            logger.warning("No context data found for the given query.")

            # Fallback Strategy
            last_assistant_response = clean_response(last_assistant_response) if last_assistant_response else ""
            small_talk = self.ca.generate_response(f"""You are a knowledgeable, friendly assistant in a natural conversation with a user. Your goal is to respond thoughtfully to the user’s query, focusing on keeping the conversation engaging and flowing naturally.

                                                    User Query: "{query}"

                                                    {f'The last thing the user asked was: "{last_user_query}".' if last_user_query else ''}
                                                    {f'Your last response was: "{last_assistant_response}".' if last_assistant_response else ''}

                                                    Guidelines:
                                                    1. **Context Awareness**: Use the context from the last message(s) only if it’s directly relevant to the new query. If the query is unrelated, respond independently without referencing previous messages.
                                                    2. **Avoid Repetition**: Don’t repeat the user's words verbatim or re-ask recent questions. If the user has already answered a conversational prompt like "How are you?", avoid asking similar questions like "How about you?" unless it naturally fits the flow.
                                                    3. **Follow-up Sensitivity**: If you asked a question in your last response and the user has replied directly (e.g., "I'm good, thanks"), acknowledge their response without re-asking similar questions.
                                                    4. **Keep it Dynamic**: Vary responses and avoid default phrases. Acknowledge the user’s answers and continue moving forward in a natural flow.

                                                    Provide a response that keeps the conversation engaging, relevant, and natural.
                                                """)

            logger.info(f"Generated small talk response: '{small_talk}'")
            return small_talk


        context = get_top_matches(context, normalized_query, top_n=1)

        node_label = ""
        if "node label" in context.columns and not context["node label"].isna().values[0]:
            node_label = context["node label"].values[0]
            logger.info(f"Node label found: {node_label}")
        else:
            logger.debug("Node label not found or is NaN.")

        entity_id = ""
        if "subject_id" in context.columns and not context["subject_id"].isna().values[0]:
            entity_id = context["subject_id"].values[0]
            logger.info(f"Entity ID found: {entity_id}")
        else:
            logger.debug("Entity ID not found or is NaN.")


        if "CURRENT MODE" == "RECOMMENDER":
            node_label = self.db.normalize_string(node_label)
            if node_label in self.db.people_names:
                movie_ids = self.db.people_movie_mapping[entity_id]
                context = self.db.fetch(movie_ids, "subject_id")
                subject_labels = context["node label"].tolist()
                return ", ".join(subject_labels)

            context.dropna(axis=1, inplace=True)
            columns = [col for col in context.columns if col in db.movie_recommender_db.columns]
            red_db = self.db.movie_recommender_db[columns]
            context = context[columns]

            # drop the identified row already in the context
            subject_id_to_remove = context["subject_id"].values[0]
            red_db = red_db[red_db["subject_id"] != subject_id_to_remove]

            red_db.dropna(thresh=len(red_db.columns) - 0, inplace=True)
            red_db.reset_index(drop=True, inplace=True)

            def calculate_similarity(i, row):
                similarities = []
                for col in context.columns:
                    if pd.isna(row[col]):
                        continue

                    if col in ["node label", "subject_id"]:
                        continue

                    similarity = fuzz.ratio(row[col], context[col].iloc[0]) / 100.0
                    similarities.append(similarity)

                return (i, np.mean(similarities) if similarities else 0)

            top_scores = []
            for i, row in red_db.iterrows():
                index, score = calculate_similarity(i, row)

                if len(top_scores) < 3:
                    heapq.heappush(top_scores, (score, index))
                else:
                    heapq.heappushpop(top_scores, (score, index))

            # Extract indices from top 3 scores
            top_indices = [index for score, index in top_scores]
            top_rows = red_db.iloc[top_indices]
            subject_labels = top_rows['node label'].tolist()
            return ", ".join(subject_labels)

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

        # Always keep these columns
        col_always_keep = ["node label"]

        combined_columns = set(top_columns_embeddings + col_always_keep)
        top_columns = [col for col in combined_columns if col in context.columns]
        logger.debug(f"Final selected columns: {top_columns}")
        filtered_context_df = context[top_columns]

        answer = self.qa.query(query, filtered_context_df)
        logger.debug(f"Answer from QA model: '{answer}'")
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