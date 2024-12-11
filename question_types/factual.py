import random

import numpy as np
import pandas as pd
import logging
import heapq
from rapidfuzz import process, fuzz

from database.database import DataBase
from models.graph_embeddings import GraphEmbeddings
from models.query_embedder import QueryEmbedderContextualized
from models.question_answering_agent import QuestionAnsweringAgent
from models.conversation_agent import ConversationAgent
from models.query_routing import QueryRouter

from sentence_transformers import util

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
    def __init__(self, sparql, crowdsourcer):
        logger.info("Initializing FactualQuestions class...")
        self.sparql = sparql
        self.qr = QueryRouter()
        logger.info("QueryRouter initialized")
        self.db = DataBase()
        logger.info("Database initialized.")
        self.qe = QueryEmbedderContextualized()
        logger.info("QueryEmbedderContextualized initialized.")
        self.qa = QuestionAnsweringAgent()
        logger.info("QuestionAnsweringAgent initialized.")
        self.ca = ConversationAgent(model_name="google/flan-t5-xl")
        logger.info("ConversationAgent initialized.")
        self.ge = GraphEmbeddings(graph=self.db.db)
        logger.info("GraphEmbeddings initialized.")
        self.crowdsourcing = crowdsourcer
        logger.info("CrowdSourcing initialized.")
        logger.info("...FactualQuestions class initialized successfully.")


    @measure_time
    def answer_query(self, query: str, last_user_query: str, last_assistant_response: str, recommender) -> str:
        logger.info(f"Query: {query.strip()}")
        normalized_query = self.db.normalize_string(query)

        ###############################################################################################################
        # QUERY ROUTING
        ###############################################################################################################
        query_route = self.qr.predict(query)
        logger.info(f"Query Routing: {query_route}")

        ###############################################################################################################
        # SMALL TALK
        ###############################################################################################################
        if query_route == "unrelated":
            small_talk = self.ca.generate_response(f"""You are a friendly and knowledgeable assistant engaged in a natural conversation. Respond to the User Query while following these guidelines:

                                                    **User Query:** "{query}"

                                                    **Guidelines:**
                                                    1. **Context Awareness:** Use previous messages only if directly relevant to the current query.
                                                    2. **Avoid Repetition:** Don’t repeat the user's words.

                                                    Provide an engaging, relevant, and natural response.
                                                """)

            logger.info(f"Generated small talk response: '{small_talk}'")
            return small_talk

        ###############################################################################################################
        # RECOMMENDATION
        ###############################################################################################################
        if query_route == "recommendation":
            recommended_movies, extracted_entities = recommender.recommend_movies(query)
            logger.info(f"Recommended movies: {', '.join([m for m, _ in recommended_movies])}")

            movies_with_details = []

            user_people = extracted_entities.get('people', set())

            for movie, score in recommended_movies:

                details = recommender.movie_details.get(movie, {
                    'director': 'Unknown',
                    'genres': 'Unknown',
                    'publication_date': 'Unknown',
                    'imdb_url': 'Not Available',
                    'cast': 'Unknown'
                })

                movies_with_details.append({
                    'title': movie,
                    'score': score,
                    'details': details
                })

            movies_list = []
            for i, movie_info in enumerate(movies_with_details):
                details = movie_info['details']
                director = details.get('director', 'Unknown')
                genres = details.get('genres', 'Unknown')
                pub_date = details.get('publication_date', 'Unknown')
                imdb_url = details.get('imdb_url', 'Not Available')
                cast = details.get('cast', '')


                genre_label = "Genres" if ',' in genres else "Genre"

                movie_entry = (
                    f"{i + 1}) {movie_info['title']}\n"
                    f"   Directed by: {director}\n"
                    f"   {genre_label}: {genres}\n"
                    f"   Release Date: {pub_date}\n"
                )

                if user_people and cast:
                    cast_members = [member.strip().lower() for member in cast.split(',')]
                    matching_cast = set(cast_members).intersection(user_people)
                    matching_cast = [cast_member.capitalize() for cast_member in matching_cast]

                    if matching_cast:
                        cast_label = "Cast Member" if len(matching_cast) == 1 else "Cast Members"
                        matching_cast_str = ', '.join(matching_cast)
                        movie_entry += f"   {cast_label}: {matching_cast_str}\n"

                movie_entry += f"   More Info: {imdb_url}\n"

                movies_list.append(movie_entry)

            movies_string = "\n".join(movies_list)

            identified_entities = set()
            for entity_set in extracted_entities.values():
                identified_entities.update(entity_set)
            entities_string = ", ".join(identified_entities) if identified_entities else "your preferences"

            formatted_recommendation = (
                f"Based on your interest in: {entities_string}\n\n"
                f"I recommend the following movies:\n\n"
                f"{movies_string}\n\n"
            )

            basis = []
            if extracted_entities.get('genres'):
                basis.append("genres")

            if extracted_entities.get('people'):
                basis.append("directors/actors")

            if extracted_entities.get('movies'):
                basis.append("similar movies")

            if basis:
                if len(basis) == 1:
                    basis_str = basis[0]
                else:
                    basis_str = ", ".join(basis[:-1]) + " and " + basis[-1]

                formatted_recommendation += f"The recommendations are based on {basis_str}.\n\n"
            else:
                formatted_recommendation += "The recommendations are based on your preferences.\n\n"

            formatted_recommendation += "Enjoy your movie time!"

            return formatted_recommendation

        ###############################################################################################################
        # Continuing for factual, crowdsourcing or multimedia
        fuzzy_person_match, person_full_match, person_match_length = fuzzy_match(
            normalized_query, self.db.people_names, self.db)
        fuzzy_movie_match, movie_full_match, movie_match_length = fuzzy_match(
            normalized_query, self.db.movie_names, self.db)

        fuzzy_person_matches, fuzzy_movie_matches = [], []

        if movie_full_match or person_full_match:
            if movie_full_match and (not person_full_match or movie_match_length > person_match_length):
                fuzzy_movie_matches = [fuzzy_movie_match]
            elif person_full_match:
                fuzzy_person_matches = [fuzzy_person_match]
        elif fuzzy_movie_match and fuzzy_person_match:
            if person_match_length > movie_match_length:
                fuzzy_movie_matches = []
            else:
                fuzzy_person_matches = []

        fuzzy_movie_context = self.db.fetch(fuzzy_movie_matches, "subject_id", normalized=False)
        fuzzy_person_context = self.db.fetch(fuzzy_person_matches, "subject_id", normalized=False)

        context = pd.concat([fuzzy_movie_context, fuzzy_person_context])

        if context.empty:
            logger.warning("No context data found for the given query using fuzzy matching.")

            entities = recommender.extract_entities(query)
            logger.info(f"Prefix Tree: {entities}")
            people_entities = list(entities.get('people', []))
            movie_entities = list(entities.get('movies', []))

            people_context = self.db.fetch(people_entities, "subject_label", normalized=False)
            movie_context = self.db.fetch(movie_entities, "subject_label", normalized=False)

            context = pd.concat([movie_context, people_context])

            if context.empty:
                logger.warning("No context data found for the given query using prefix tree.")
                # Fallback Strategy
                small_talk = self.ca.generate_response(f"""You are a friendly and knowledgeable assistant engaged in a natural conversation. Respond to the User Query while following these guidelines:
        
                                                        **User Query:** "{query}"
        
                                                        **Guidelines:**
                                                        1. **Context Awareness:** Use previous messages only if directly relevant to the current query.
                                                        2. **Avoid Repetition:** Don’t repeat the user's words.
    
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

        ###############################################################################################################
        # MULTIMEDIA
        ###############################################################################################################
        if query_route == "multimedia":
            if "imdb id" in context.columns and not context["imdb id"].isna().values[0]:
                logger.info("Detected multimedia query.")
                imdb_id = context["imdb id"].values[0]
                image = self.db.get_image(imdb_id, is_movie=True if fuzzy_movie_matches else False)
                if image and node_label:
                    return f"Here is a Picture of {node_label}\n image:{image}"
                return f"Sorry, I don't find any image for {node_label}."

        ###############################################################################################################
        # FACTUAL
        ###############################################################################################################

        # CROWDSOURCING
        crowd_response = self.crowdsourcing.classify_query(query)

        if crowd_response != "factual":
            return crowd_response
        ###############################################################################################################
        # CONTINUE WITH FACTUAL ANSWERING
        # Remove unused columns
        elements_to_remove = ["image", "color", "sport"]
        context = context.drop(columns=elements_to_remove, errors='ignore')

        # Initial context for embeddings where original column names are required
        initial_context = context.copy()

        # Rename columns
        columns_to_rename = {
            "cast member": "movie cast",
            "notable work": "acted in"
        }
        context = context.rename(columns={k: v for k, v in columns_to_rename.items() if k in context.columns})

        columns_to_duplicate = [("acted in", "played in"),
                                ("acted in", "appeared in"),
                                ("movie cast", "actors"),
                                ("movie cast", "players")]

        for col_to_duplicate, col in columns_to_duplicate:
            if col_to_duplicate in context.columns:
                context[col] = context[col_to_duplicate]

        context.dropna(axis=1, inplace=True)

        query_filtered = filter_query(query, node_label)

        column_embeddings = {col: self.qe.embed_phrase(col) for col in context.columns}
        query_embeddings = [self.qe.embed_phrase(word) for word in query_filtered.split()]
        top_columns_embeddings = find_closest_columns(query_embeddings, column_embeddings)

        # Always keep these columns
        col_always_keep = ["node label"]

        combined_columns = set(top_columns_embeddings + col_always_keep)
        top_columns = [col for col in combined_columns if col in context.columns]
        filtered_context_df = context[top_columns]

        answer = self.qa.query(query, filtered_context_df)
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

