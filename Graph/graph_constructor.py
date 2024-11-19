import igraph as ig
import networkx as nx
from tqdm import tqdm

def construct_graph(db):
    """
    Constructs a NetworkX graph from the provided dataset.

    Parameters:
    - db: pandas DataFrame containing the dataset with columns:
        - 'predicate_label'
        - 'object_label'
        - 'subject_label'

    Returns:
    - G_nx: NetworkX Graph object
    """
    # Relevant predicates
    relevant_predicates = [
        "director",
        "performer",
        "genre",
        "screenwriter",
        "cast member",
        "publication date",
        "mpaa film rating"
    ]

    db_filtered = db[db.predicate_label.isin(relevant_predicates)].copy()
    db_filtered['object_label'] = db_filtered['object_label'].astype(str)

    db_filtered.loc[db_filtered['predicate_label'] == "publication date", 'object_label'] = (
        db_filtered.loc[db_filtered['predicate_label'] == "publication date", 'object_label']
        .apply(lambda x: f"{(int(x.split('-')[0]) // 10) * 10}-{(int(x.split('-')[0]) // 10) * 10 + 9}"
        if x.split('-')[0].isdigit() else None)

    # Initialize igraph Graph
    G = ig.Graph(directed=False)
    node_to_index = {}
    index_to_node = []

    def get_or_add_node(node_label):
        if node_label not in node_to_index:
            index = len(index_to_node)
            node_to_index[node_label] = index
            index_to_node.append(node_label)
            G.add_vertex(name=node_label)
        return node_to_index[node_label]

    # Prepare the data
    df = db_filtered.copy()
    df['object_label'] = df['object_label'].str.split(',')
    df = df.explode('object_label')
    df['object_label'] = df['object_label'].str.strip()

    # Build edges
    edge_dict = {}

    for _, row in tqdm(df.iterrows(), desc="Graph construction", total=len(df)):
        label = row['predicate_label']
        individual_value = row['object_label']
        movie = row['subject_label']

        value_index = get_or_add_node(individual_value)
        movie_index = get_or_add_node(movie)

        edge = (movie_index, value_index)

        if edge in edge_dict:
            edge_dict[edge] += 1
        else:
            edge_dict[edge] = 1

    # Add edges to the graph
    edges_to_add = list(edge_dict.keys())
    weights = list(edge_dict.values())

    G.add_edges(edges_to_add)
    G.es["weight"] = weights

    # Convert igraph to NetworkX
    G_nx = nx.Graph()

    for vertex in G.vs:
        G_nx.add_node(vertex["name"])

    for edge in G.es:
        source = edge.source
        target = edge.target
        weight = edge["weight"]
        source_label = G.vs[source]["name"]
        target_label = G.vs[target]["name"]
        G_nx.add_edge(source_label, target_label, weight=weight)

    return G_nx