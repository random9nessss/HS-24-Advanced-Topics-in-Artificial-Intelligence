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
from models.query_routing import QueryRouter

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
    def __init__(self, sparql):
        logger.info("Initializing FactualQuestions class...")
        self.sparql = sparql
        self.qr = QueryRouter()
        logger.info("QueryRouter initialized")
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
        logger.info("...FactualQuestions class initialized successfully.")


    @measure_time
    def answer_query(self, query: str, last_user_query: str, last_assistant_response: str, recommender) -> str:
        logger.info(f"Query: {query.strip()}")
        normalized_query = self.db.normalize_string(query)

        ###############
        # ROUTING
        ###############
        query_route = self.qr.predict(query)
        logger.info(f"Query Routing: {query_route}")

        ###############
        # SMALL TALK
        ###############
        if query_route == "unrelated":
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

        ########################
        # RECOMMENDATION MODULE
        ########################
        if query_route == "recommendation":
            recommended_movies, extracted_entities = recommender.recommend_movies(query)
            logger.info(f"Recommended movies: {', '.join([m for m, _ in recommended_movies])}")

            movies_with_details = []

            user_people = extracted_entities.get('people', set())

            for movie, score in recommended_movies:
                wikidata_id = self.db.get_movie_wikidata_id(movie)
                logger.info(f"Retrieved Wikidata ID for '{movie}': {wikidata_id}")

                details = {}
                if wikidata_id:
                    try:
                        details = self.sparql.get_movie_details(wikidata_id)
                        #logger.info(f"Fetched details for '{movie}': {details}")

                    except Exception as e:
                        logger.error(f"Error fetching details for '{movie}' (Wikidata ID: {wikidata_id}): {e}")

                else:
                    logger.warning(f"Missing Wikidata ID for '{movie}'. Skipping details fetch.")

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
                f"{movies_string}\n"
                "Enjoy your movie time!"
            )

            return formatted_recommendation

        fuzzy_person_match, person_full_match, person_match_length = fuzzy_match(normalized_query, self.db.people_names,
                                                                                 self.db)
        fuzzy_movie_match, movie_full_match, movie_match_length = fuzzy_match(normalized_query, self.db.movie_names,
                                                                              self.db)

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

        ###############
        # MULTIMEDIA
        ###############
        if query_route == "multimedia":
            if "imdb id" in context.columns and not context["imdb id"].isna().values[0]:
                logger.info("Detected multimedia query.")
                imdb_id = context["imdb id"].values[0]
                image = self.db.get_image(imdb_id, is_movie=True if fuzzy_movie_matches else False)
                if image and node_label:
                    return f"Here is a Picture of {node_label}\n image:{image}"
                return f"Sorry, I don't find any image for {node_label}."

        ###############
        # FACTUAL
        ###############

        # Remove unused columns
        elements_to_remove = ["image", "color", "sport"]
        context = context.drop(columns=elements_to_remove, errors='ignore')

        # Initial context for embeddings where original column names are required
        initial_context = context.copy()


        ###############
        # CROWD SOURCING
        ###############
        corrections = self.db.crowd_data.get(self.db.normalize_string(node_label), [])
        reject, support, inter_aggreement = 0, 0, 0
        replaced_object, replaced_predicate = "", ""
        for orig_predicate, orig_object, corrected_predicate, corrected_object, voted_correct, voted_incorrect, aggreement in corrections:
            logger.info(f"Original: {orig_predicate} - {orig_object}, Corrected: {corrected_predicate} - {corrected_object}")

            logger.info(f"Set predicate '{corrected_predicate}' to '{corrected_object}'")
            context[corrected_predicate] = corrected_object

            if orig_predicate != corrected_predicate and orig_predicate in context.columns:
                logger.info(f"Deleted predicate '{orig_predicate}' from context")
                context.drop(columns=[orig_predicate], inplace=True)

            replaced_predicate = corrected_predicate
            replaced_object = corrected_object
            reject, support, inter_aggreement = voted_incorrect, voted_correct, aggreement
        ###############
        # CROWD SOURCING - END
        ###############

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

        ###############
        # CROWD SOURCING - We should replace this with a more sophisticated approach
        # If the corrected predicate or object is in the answer, we should add a comment
        # With numbers I experienced here some issues with LLM, it did hallucinate
        ###############
        crowd_source_comment = ""
        if reject > support and (replaced_predicate or replaced_object) and (replaced_predicate in formatted_answer or replaced_object in formatted_answer):
            crowd_source_comment = f"\n[Crowd, inter-rater agreement {inter_aggreement}, The answer distribution for this specific task was {support} support votes, {reject} reject votes]"
        ###############
        # CROWD SOURCING - END
        ###############

        return f"Graph:\n{formatted_answer}{crowd_source_comment}\n\nEmbeddings:\n{embedding_answer}"