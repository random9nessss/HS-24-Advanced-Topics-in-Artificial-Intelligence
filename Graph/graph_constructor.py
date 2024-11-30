import igraph as ig
import networkx as nx
from tqdm import tqdm
import pandas as pd

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

    # Selected Predicates
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
    )

    db_filtered.loc[db_filtered['predicate_label'] == "genre", 'object_label'] = (
        db_filtered.loc[db_filtered['predicate_label'] == "genre", 'object_label']
        .str.replace("film", "", regex=False).str.strip()
    )

    animated_rows = db_filtered[
        (db_filtered['predicate_label'] == "genre") &
        (db_filtered['object_label'] == "animated")
        ].copy()
    animated_rows['object_label'] = "animation"

    db_filtered = pd.concat([db_filtered, animated_rows], ignore_index=True)

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

    df = db_filtered.copy()
    df['object_label'] = df['object_label'].str.split(',')
    df = df.explode('object_label')
    df['object_label'] = df['object_label'].str.strip()

    edge_dict = {}
    edge_predicates = {}

    # Weighted Predicates
    predicate_weights = {
        "director": 5,
        "genre": 5,
        "screenwriter": 3,
        "cast member": 1,
        "performer": 2,
        "publication date": 2,
        "mpaa film rating": 2
    }

    for _, row in tqdm(df.iterrows(), desc="Graph construction", total=len(df)):
        label = row['predicate_label']
        individual_value = row['object_label']
        movie = row['subject_label']

        value_index = get_or_add_node(individual_value)
        movie_index = get_or_add_node(movie)

        edge = (movie_index, value_index)
        weight = predicate_weights.get(label, 1)

        if edge in edge_dict:
            edge_dict[edge] += weight
        else:
            edge_dict[edge] = weight
            edge_predicates[edge] = label

    edges_to_add = list(edge_dict.keys())
    weights = list(edge_dict.values())
    predicates = [edge_predicates[edge] for edge in edges_to_add]

    G.add_edges(edges_to_add)
    G.es["weight"] = weights
    G.es["predicate_label"] = predicates

    G_nx = nx.Graph()

    for vertex in G.vs:
        G_nx.add_node(vertex["name"])

    for edge in G.es:
        source = edge.source
        target = edge.target
        weight = edge["weight"]
        predicate_label = edge["predicate_label"]
        source_label = G.vs[source]["name"]
        target_label = G.vs[target]["name"]
        G_nx.add_edge(
            source_label,
            target_label,
            weight=weight,
            predicate_label=predicate_label
        )

    return G_nx