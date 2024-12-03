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
        "mpaa film rating",
        "production company",
        "followed by",
        "follows",
        "imdb id"
    ]

    db_filtered = db[db.predicate_label.isin(relevant_predicates)].copy()

    # Added to just select movies
    imdb_rows = db_filtered[
        (db_filtered['predicate_label'] == 'imdb id') & db_filtered['object_label'].str.startswith('tt')]
    valid_subject_ids = set(imdb_rows['subject_id'])
    db_filtered = db_filtered[db_filtered['subject_id'].isin(valid_subject_ids)]

    db_filtered['object_label'] = db_filtered['object_label'].astype(str)

    selected_subject_ids = set()
    movie_release_years = {}

    for title, group in db_filtered.groupby('subject_label'):

        group_with_imdb = group[group['predicate_label'] == 'imdb id']

        if not group_with_imdb.empty:
            tt_ids = group_with_imdb[
                group_with_imdb['subject_id'].str.startswith('tt')]  # Addded to just have movie titles

            if not tt_ids.empty:
                preferred_id = tt_ids['subject_id'].iloc[0]

            else:
                preferred_id = group_with_imdb['subject_id'].iloc[0]
        else:

            group_with_pub_date = group[group['predicate_label'] == 'publication date']
            if not group_with_pub_date.empty:

                group_with_pub_date['year'] = group_with_pub_date['object_label'].apply(
                    lambda x: int(x.split('-')[0]) if x.split('-')[0].isdigit() else 0
                )
                preferred_id = group_with_pub_date.sort_values('year', ascending=False)['subject_id'].iloc[0]
            else:
                preferred_id = group['subject_id'].iloc[0]

        selected_subject_ids.add(preferred_id)

        pub_dates = group[group['subject_id'] == preferred_id][group['predicate_label'] == 'publication date'][
            'object_label']
        if not pub_dates.empty:
            year_str = pub_dates.iloc[0].split('-')[0]
            if year_str.isdigit():
                movie_release_years[title] = int(year_str)

    db_filtered = db_filtered[db_filtered['subject_id'].isin(selected_subject_ids)].copy()

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

    db_filtered = db_filtered[db_filtered.predicate_label != "imdb id"]

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
        "director": 6,
        "genre": 7,
        "screenwriter": 4,
        "cast member": 3,
        "performer": 2,
        "publication date": 2,
        "mpaa film rating": 2,
        "production company": 3,
        "followed by": 5,
        "follows": 5
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

    return G_nx, movie_release_years