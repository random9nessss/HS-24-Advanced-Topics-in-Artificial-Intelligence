import logging

from question_types.sparql import SparqlQueries
from question_types.factual import FactualQuestions
from question_types.recommendation_two import Recommender
from speakeasypy.openapi.client.rest import logger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(funcName)s | %(message)s'
)


class AgentService:

    def __init__(self):

        self.__recommender = Recommender(
            db_path='../dataset/extended_graph_triples.pkl',
            movie_data_path='../dataset/movie_db.json',
            people_data_path='../dataset/people_db.json'
        )

        self.__factual = FactualQuestions()

        self.__sparql = SparqlQueries("../dataset/14_graph.nt")

        logger.info("READY TO ANSWER QUESTIONS")

    def react(self, message: str, last_message_user: str, last_message_agent: str) -> str:
        try:
            reaction = self.execute_sparql(message)
        except Exception as e:
            reaction = self.execute_factual(message, last_message_user, last_message_user)
            return reaction
        return reaction

    def execute_sparql(self, query: str) -> str:
        sparql_result = self.__sparql.execute_query(query)
        result_lst = [
            str(item) for row in sparql_result
            for item in (row if isinstance(row, tuple) else [row])
        ]
        result_str = ", ".join(result_lst)

        if len(result_lst) == 1:
            return f"Here is the result for your query: {result_str}"
        elif len(result_lst) > 1:
            return f"Here are the results for your query: {result_str}"
        return "Your query did not match anything."

    def execute_factual(self, query: str, last_message_user: str, last_message_agent: str) -> str:
        answer = self.__factual.answer_query(query, last_message_user, last_message_agent, self.__recommender)
        return answer

    def test_queries(self):

        queries = [
            '''
            prefix wdt: <http://www.wikidata.org/prop/direct/>
            prefix wd: <http://www.wikidata.org/entity/>

            SELECT ?obj ?lbl WHERE {
                ?ent rdfs:label "Jean Van Hamme"@en .
                ?ent wdt:P106 ?obj .
                ?obj rdfs:label ?lbl .
            }
            ''',
            '''
            PREFIX ddis: <http://ddis.ch/atai/>   
            PREFIX wd: <http://www.wikidata.org/entity/>   
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>   
            PREFIX schema: <http://schema.org/>   
            
            SELECT ?lbl WHERE {  
                ?movie wdt:P31 wd:Q11424 .  
                ?movie ddis:rating ?rating .  
                ?movie rdfs:label ?lbl .  
            }
            ORDER BY DESC(?rating)   
            LIMIT 1 
            ''',
            '''
            PREFIX ddis: <http://ddis.ch/atai/>   
            PREFIX wd: <http://www.wikidata.org/entity/>   
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>   
            PREFIX schema: <http://schema.org/>   
            
            SELECT ?lbl WHERE {  
                ?movie wdt:P31 wd:Q11424 .  
                ?movie ddis:rating ?rating .  
                ?movie rdfs:label ?lbl .  
            }  
            ORDER BY ASC(?rating)   
            LIMIT 1 
            ''',
            '''
            PREFIX ddis: <http://ddis.ch/atai/>   
            PREFIX wd: <http://www.wikidata.org/entity/>   
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>   
            PREFIX schema: <http://schema.org/>   

            SELECT ?director WHERE {  
                ?movie rdfs:label "Apocalypse Now"@en .  
                ?movie wdt:P57 ?directorItem . 
                ?directorItem rdfs:label ?director . 
            }  
            LIMIT 1
            '''
        ]

        expected_results = [
            "Butcher",
            "Forrest Gump",
            "Vampire Assassin",
            "Francis Ford Coppola"
        ]

        for query, expected_result in zip(queries, expected_results):
            result = self.execute_sparql(query)
            if result != expected_result:
                logging.warning(f"Test failed: expected '{expected_result}', got '{result}'")