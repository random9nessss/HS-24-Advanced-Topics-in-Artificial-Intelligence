import rdflib
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(funcName)s | %(message)s'
)

class BColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class SparqlQueries:
    """
    A class to perform SPARQL queries on an RDF graph using the rdflib library.

    Attributes
    ----------
    graph : rdflib.Graph
        The RDF graph to which SPARQL queries will be applied.
    header : str
        The SPARQL query prefixes built from the predefined schemas.

    Methods
    -------
    execute_query(query: str) -> rdflib.query.Result:
        Executes the given SPARQL query on the RDF graph.
    """

    def __init__(self, file_path: str):
        """
        Initializes the SparqlQueries class by setting up an empty RDF graph
        and predefined SPARQL prefixes.
        """
        self.graph = rdflib.Graph()
        logging.info("Initializing SparqlQueries class...")
        self.graph.parse(file_path, format='turtle')
        logging.info(f"Graph parsed")
        logging.info("...SparqlQueries class initialized successfully.")

        self.schema_ddis = "<http://ddis.ch/atai/>"
        self.schema_wd = "<http://www.wikidata.org/entity/>"
        self.schema_wdt = "<http://www.wikidata.org/prop/direct/>"
        self.schema_schema = "<http://schema.org/>"

        # Builds the prefix for the SPARQL queries
        self.header = self._build_header()

    def _build_header(self) -> str:
        """
        Builds the SPARQL query prefixes using the predefined schema URIs.

        Returns
        -------
        str
            A formatted string containing the SPARQL query prefixes.
        """
        header = "\n".join([
            f"PREFIX {schema.replace('schema_', '')}: {getattr(self, schema)}"
            for schema in self.__dict__ if schema.startswith("schema_")
        ])
        return header

    def execute_query(self, query: str) -> rdflib.query.Result:
        """
        Executes the given SPARQL query on the RDF graph.

        Parameters
        ----------
        query : str
            The SPARQL query string to be executed.

        Returns
        -------
        rdflib.query.Result
            The result of the SPARQL query.
        """
        full_query = f"""
                        {self.header if 'PREFIX' not in query else ''}

                        {query}
                    """

        result = self.graph.query(full_query)
        return result

    def get_movie_details(self, wikidata_id):
        """
        Retrieve director, genre, publication date, IMDb URL, and cast members for a movie using its Wikidata ID.

        Args:
            wikidata_id (str): The Wikidata ID of the movie.

        Returns:
            dict: A dictionary containing the movie details.
        """

        wikidata_id = wikidata_id.replace("http://www.wikidata.org/entity/", "")

        query = f"""
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX ddis: <http://ddis.ch/atai/>
    
        SELECT ?director ?publicationDate ?imdbURL
               (GROUP_CONCAT(DISTINCT ?genreLabel; separator=", ") AS ?genres)
               (GROUP_CONCAT(DISTINCT ?castMemberLabel; separator=", ") AS ?cast)
        WHERE {{
            wd:{wikidata_id} wdt:P57 ?directorItem .
            ?directorItem rdfs:label ?director .
            wd:{wikidata_id} wdt:P577 ?publicationDate .
            wd:{wikidata_id} wdt:P136 ?genre .
            ?genre rdfs:label ?genreLabel .
            wd:{wikidata_id} wdt:P161 ?castMember .
            ?castMember rdfs:label ?castMemberLabel .
            OPTIONAL {{ wd:{wikidata_id} wdt:P345 ?imdbID . }}
            BIND(CONCAT("https://www.imdb.com/title/", ?imdbID) AS ?imdbURL)
            FILTER (lang(?genreLabel) = "en")
            FILTER (lang(?castMemberLabel) = "en")
            FILTER (lang(?director) = "en")
        }}
        GROUP BY ?director ?publicationDate ?imdbURL
        """
        try:
            result = self.execute_query(query)
            details = {}

            for row in result:
                details['director'] = str(row.director) if hasattr(row, 'director') else 'Unknown'
                details['genres'] = str(row.genres) if hasattr(row, 'genres') else 'Unknown'
                details['publication_date'] = str(row.publicationDate) if hasattr(row, 'publicationDate') else 'Unknown'
                details['imdb_url'] = str(row.imdbURL) if hasattr(row, 'imdbURL') else 'Not Available'
                details['cast'] = str(row.cast) if hasattr(row, 'cast') else 'Unknown'
            return details

        except Exception as e:
            logging.error(f"SPARQL query failed for Wikidata ID {wikidata_id}: {e}")

            return {
                'director': 'Unknown',
                'genres': 'Unknown',
                'publication_date': 'Unknown',
                'imdb_url': 'Not Available',
                'cast': 'Unknown'
            }