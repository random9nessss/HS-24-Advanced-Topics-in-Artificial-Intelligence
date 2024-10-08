import rdflib


def create_minimal_rdf_by_divisor(input_file: str, output_file: str, format: str, divisor: int):
    """
    Creates a minimal version of the RDF graph by selecting a fraction of triples
    based on the divisor value.

    Parameters
    ----------
    input_file : str
        The path to the original RDF file.
    output_file : str
        The path to save the minimal RDF file.
    format : str, optional
        The format of the input and output RDF files, by default 'nt'.
        Use 'nt' for N-Triples format (e.g., '.nt' files).
    divisor : int, optional
        The divisor to determine the fraction of triples to include, by default 4.
        A divisor of 4 means 1/4 of the original size.
    """
    # Load the full RDF graph
    graph = rdflib.Graph()
    graph.parse(input_file, format=format)

    # Calculate the number of triples to include
    total_triples = len(graph)
    max_triples = max(1, total_triples // divisor)  # Ensure at least 1 triple is added

    # Create a new graph for the minimal version
    minimal_graph = rdflib.Graph()

    # Add up to `max_triples` triples to the minimal graph
    for i, triple in enumerate(graph):
        if i >= max_triples:
            break
        minimal_graph.add(triple)

    # Save the minimal graph to a new file in `.nt` format
    minimal_graph.serialize(destination=output_file, format='turtle')

# Example usage
create_minimal_rdf_by_divisor(
    input_file="./14_graph.nt",
    output_file="./minimal_graph.nt",
    format='turtle',
    divisor=15
)
