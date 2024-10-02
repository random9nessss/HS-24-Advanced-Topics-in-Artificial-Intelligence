from rdflib.namespace import Namespace, RDFS
from rdflib.term import URIRef, Literal
import csv
import pandas as pd
import rdflib

from flask import Flask, request
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Increase logging level
app.logger.setLevel(logging.DEBUG)

class Dataset:

    def __init__(self):
        self.__graph = self.__init_graph()

    def __init_graph(self):

        graph = rdflib.Graph()
        graph.parse('../dataset/14_graph.nt', format='turtle')

        """
        WD = Namespace('http://www.wikidata.org/entity/')
        WDT = Namespace('http://www.wikidata.org/prop/direct/')
        SCHEMA = Namespace('http://schema.org/')
        DDIS = Namespace('http://ddis.ch/atai/')


        entities = set(graph.subjects()) | {s for s in graph.objects() if isinstance(s, URIRef)}
        predicates = set(graph.predicates())
        literals = {s for s in graph.objects() if isinstance(s, Literal)}
        with_type = set(graph.subjects(WDT['P31'], None))
        with_super = set(graph.subjects(WDT['P279'], None))
        types = set(graph.objects(None, WDT['P31']))
        supers = set(graph.objects(None, WDT['P279']))
        with_label = set(graph.subjects(RDFS.label, None))

        n_ents = len(entities)
        n_rels = len(predicates)
        n_lits = len(literals)
        t_tot = len(graph)
        t_ent = len([1 for s,p,o in graph.triples((None, None, None)) if isinstance(o, URIRef)])
        t_lit = t_tot - t_ent
        n_notype = len(entities - with_type - with_super)
        n_notype_flt = len(entities - with_type - with_super - types - supers)

        pd.DataFrame([
            ('number of entities', f'{n_ents:n}'),
            ('number of literals', f'{n_lits:n}'),
            ('number of predicates', f'{n_rels:n}'),
            ('number of triples', f'{t_tot:n}'),
            ('number of ent-ent triples', f'{t_ent:n}'),
            ('number of ent-lit triples', f'{t_lit:n}'),
            ('number of entities w/o label', f'{len(entities - with_label):n}'),
            ('number of predicates w/o label', f'{len(predicates - with_label):n}'),
            ('number of entities w/o type', f'{n_notype:n}'),
            ('number of instances w/o type', f'{n_notype_flt:n}'),
            ])

        top250 = set(open('../dataset/imdb-top-250.t').read().split('\n')) - {''}

        ent_lit_preds = {p for s,p,o in graph.triples((None, None, None)) if isinstance(o, Literal)}

        roots = {
            WD['Q8242']:        'literature',
            WD['Q5']:           'human',
            WD['Q483394']:      'genre',
            WD['Q95074']:       'character',
            WD['Q11424']:       'film',
            WD['Q15416']:       'tv',
            WD['Q618779']:      'award',
            WD['Q27096213']:    'geographic',
            WD['Q43229']:       'organisation',
            WD['Q34770']:       'language',
            WD['Q7725310']:     'series',
            WD['Q47461344']:    'written work',
        }
        """
        print("Stage 3 - SURVIVED")
        return graph

    def execute_sparql(self, query: str) -> list[str]:
        print("Received Query: ", query)
        result = [str(s) for s, in self.__graph.query(query)]
        print(result)
        return result

dataset = Dataset()

@app.route('/sparql', methods=["POST"])
def sparql():
    return dataset.execute_sparql(request.json.get('query'))


if __name__ == "__main__":
    app.run(port=5000, debug=True, host='0.0.0.0')


"""
# top user-rated movies
[str(s) for s, in graph.query('''
    PREFIX ddis: <http://ddis.ch/atai/> 
    PREFIX wd: <http://www.wikidata.org/entity/> 
    PREFIX wdt: <http://www.wikidata.org/prop/direct/> 
    PREFIX schema: <http://schema.org/> 

    SELECT ?lbl WHERE {
        SELECT ?movie ?lbl ?rating WHERE {
            ?movie wdt:P31 wd:Q11424 .
            ?movie ddis:rating ?rating .
            ?movie rdfs:label ?lbl .
        }
        ORDER BY DESC(?rating) 
        LIMIT 20
    }
    ''')]

# bottom user-rated movies
[str(s) for s, in graph.query('''
    PREFIX ddis: <http://ddis.ch/atai/> 
    PREFIX wd: <http://www.wikidata.org/entity/> 
    PREFIX wdt: <http://www.wikidata.org/prop/direct/> 
    PREFIX schema: <http://schema.org/> 

    SELECT ?lbl WHERE {
        SELECT ?movie ?lbl ?rating WHERE {
            ?movie wdt:P31 wd:Q11424 .
            ?movie ddis:rating ?rating .
            ?movie rdfs:label ?lbl .
        }
        ORDER BY ASC(?rating) 
        LIMIT 10
    }
    ''')]

# some info about a Apocalypse Now

header = '''
    PREFIX ddis: <http://ddis.ch/atai/>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX schema: <http://schema.org/>
'''

tuple_list = list(graph.query(header + '''
    SELECT * WHERE {
        ?movie rdfs:label "Apocalypse Now"@en .
        ?movie wdt:P57/rdfs:label ?director .
        OPTIONAL { ?movie ddis:rating ?rating } .
        OPTIONAL { ?movie wdt:P577 ?value}
    }
    '''))

first_tuple = tuple_list[0]

print(f"First tuple: {first_tuple}")
print('------------')

for elements in first_tuple:
    print(elements)

# dealing with optional parameters
tuple_list = list(graph.query(header + '''
    SELECT ?lbl ?rating WHERE {
        ?movie rdfs:label ?lbl .
        ?movie wdt:P57/rdfs:label ?director .
        OPTIONAL { ?movie ddis:rating ?rating } .
        OPTIONAL { ?movie wdt:P577 ?value}
    }
    LIMIT 10
    '''))

# unpacking the tuple in the loop
for (movie_label, rating) in tuple_list:
    if rating:
        print(f"{movie_label} has a rating of {rating} ⭐️")
    else:
        print(f"{movie_label} has no rating 😔")

# all movies directed by Terry Gilliam
[str(s) for s, in graph.query('''
    PREFIX ddis: <http://ddis.ch/atai/> 
    PREFIX wd: <http://www.wikidata.org/entity/> 
    PREFIX wdt: <http://www.wikidata.org/prop/direct/> 
    PREFIX schema: <http://schema.org/> 

    SELECT ?lbl WHERE {
        ?director rdfs:label "Terry Gilliam"@en .
        ?movie wdt:P57 ?director .
        ?movie rdfs:label ?lbl
    }
    ''')]

# neo-noir movies featuring Ryan Gosling
[str(s) for s, in graph.query('''
    PREFIX ddis: <http://ddis.ch/atai/> 
    PREFIX wd: <http://www.wikidata.org/entity/> 
    PREFIX wdt: <http://www.wikidata.org/prop/direct/> 
    PREFIX schema: <http://schema.org/> 

    SELECT ?lbl WHERE {
        ?genre rdfs:label "neo-noir"@en .
        ?actor rdfs:label "Ryan Gosling"@en .
        ?movie wdt:P136 ?genre .
        ?movie wdt:P161 ?actor .
        ?movie rdfs:label ?lbl .
    }
    ''')]

# movies with largest cast member list
[(str(s), int(nc)) for s, nc in graph.query('''
    PREFIX ddis: <http://ddis.ch/atai/> 
    PREFIX wd: <http://www.wikidata.org/entity/> 
    PREFIX wdt: <http://www.wikidata.org/prop/direct/> 
    PREFIX schema: <http://schema.org/> 

    SELECT ?lbl ?nc WHERE {
        SELECT ?movie ?lbl (count(?cast) as ?nc) WHERE {
            ?movie wdt:P31 wd:Q11424 .
            ?movie rdfs:label ?lbl .
            ?movie wdt:P161 ?cast .
        }
        GROUP BY ?movie
        ORDER BY DESC(?nc)
        LIMIT 10
    }
    ''')]

# cast of Moon
[str(s) for s, in graph.query('''
    PREFIX ddis: <http://ddis.ch/atai/> 
    PREFIX wd: <http://www.wikidata.org/entity/> 
    PREFIX wdt: <http://www.wikidata.org/prop/direct/> 
    PREFIX schema: <http://schema.org/> 

    SELECT ?lbl WHERE {
        ?movie rdfs:label "Moon"@en .
        ?movie wdt:P161 ?cast .
        ?cast rdfs:label ?lbl .
    }
    ''')]


# winners of Cannes best movie (Palme d'Or)
a = [(str(d), str(s)) for s, d in graph.query(header + '''
    SELECT ?lbl ?pubdate WHERE {
        ?award rdfs:label "Palme d'Or"@en .
        ?movie wdt:P166 ?award .
        ?movie rdfs:label ?lbl .
        ?movie wdt:P577 ?pubdate .
        FILTER (?pubdate > "2011-01-01"^^xsd:date)
    }
    ORDER BY DESC(?pubdate)
    ''')]

# this can be also written as (notice the ";"):
b = [(str(d), str(s)) for s, d in graph.query(header + '''
    SELECT ?lbl ?pubdate WHERE {
      ?award rdfs:label "Palme d'Or"@en.
      ?movie wdt:P166 ?award; rdfs:label ?lbl; wdt:P577 ?pubdate.
      FILTER(?pubdate > "2011-01-01"^^xsd:date)
    }
    ORDER BY DESC (?pubdate)
    ''')]

assert (a == b)
print(a)
"""


