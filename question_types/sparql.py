import rdflib
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
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
        logging.info(f"{BColors.OKGREEN}Parsing graph{BColors.ENDC}")
        self.graph.parse(file_path, format='turtle')
        logging.info(f"{BColors.OKGREEN}Graph parsed{BColors.ENDC}")

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


# sparql = SparqlQueries(file_path="./movie-bot/dataset/14_graph.nt")
