import logging

from question_types.sparql import SparqlQueries

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)


class AgentService:

    def __init__(self):
        # self.__sparql = SparqlQueries("../dataset/minimal_graph.nt")
        self.__sparql = SparqlQueries("../dataset/14_graph.nt")
        self.test_queries()

    def react(self, message: str) -> str:
        self.execute_sparql(message)
        return f"Here is the result for your query:\n{message}"

    def execute_sparql(self, query: str) -> str:
        sparql_result = self.__sparql.execute_query(query)
        result_lst = [
            str(item) for row in sparql_result
            for item in (row if isinstance(row, tuple) else [row])
        ]
        result_str = ", ".join(result_lst)
        logging.info(f"Query result: {result_str}")
        return result_str

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
