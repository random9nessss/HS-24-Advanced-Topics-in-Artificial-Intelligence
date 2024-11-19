import csv
import rdflib
import numpy as np
import json
import pandas as pd
from sklearn.metrics import pairwise_distances
from utils.utils import (
    logger
)

class GraphEmbeddings:

    def __init__(self, graph):

        self.RDFS = rdflib.namespace.RDFS
        self.WD = rdflib.Namespace('http://www.wikidata.org/entity/')

        self.entity_emb = np.load('../development/exports/entity_embeds.npy')
        self.relation_emb = np.load('../development/exports/relation_embeds.npy')

        with open('../development/exports/entity_ids.del', 'r') as ifile:
            self.ent2id = {rdflib.term.URIRef(ent): int(idx) for idx, ent in csv.reader(ifile, delimiter='\t')}
            self.id2ent = {v: k for k, v in self.ent2id.items()}

        with open('../development/exports/relation_ids.del', 'r') as ifile:
            self.rel2id = {rdflib.term.URIRef(rel): int(idx) for idx, rel in csv.reader(ifile, delimiter='\t')}

        with open("../development/exports/predicate_db.json", encoding="utf-8") as f:
            self.predicates_db = json.load(f)

        self.ent2lbl = {rdflib.term.URIRef(row.subject_id): row.subject_label for index, row in graph.iterrows()}

    def answer_query_embedding(self, context, top_columns):
        STD_ERROR = "Could not generate an answer."
        try:
            context_id = context.subject_id.values[0]
            if not context_id:
                logger.info("No context ID found.")
                return STD_ERROR

            e_id = self.ent2id.get(rdflib.term.URIRef(context_id))
            if e_id is None:
                return STD_ERROR

            head = self.entity_emb[e_id]
            wiki_predicate_id = ""

            column_mapping = {
                "movie cast": "cast member",
                "acted in": "notable work",
                "played in": "notable work",
                "appeared in": "notable work",
                "actors": "cast member",
                "players": "cast member"
            }

            for col in top_columns:

                col = column_mapping.get(col, col)

                if col == "node label":
                    continue

                if not context[col].values[0]:
                    continue

                if col in self.predicates_db:
                    wiki_id = rdflib.term.URIRef(self.predicates_db[col])
                    if wiki_id in self.rel2id:
                        wiki_predicate_id = self.predicates_db[col]
                        break
                    else:
                        logger.info(f"Predicate {col} not found in rel2id.")
                else:
                    logger.info(f"Predicate {col} not found in predicate_db.")

            if not wiki_predicate_id:
                logger.info("No valid predicate found.")
                return STD_ERROR

            r_id = self.rel2id[rdflib.term.URIRef(wiki_predicate_id)]
            pred = self.relation_emb[r_id]

            lhs = head + pred
            dist = pairwise_distances(lhs.reshape(1, -1), self.entity_emb).reshape(-1)
            most_likely = dist.argsort()

            num_results = min(3, len(most_likely))
            results_lst = [
                (self.id2ent[idx][len(self.WD):], self.ent2lbl.get(self.id2ent[idx], ""), dist[idx], rank + 1)
                for rank, idx in enumerate(most_likely[:num_results])
            ]
            results_df = pd.DataFrame(results_lst, columns=('Entity', 'Label', 'Score', 'Rank'))

            if results_df.empty:
                logger.info("No results in embeddings found.")
                return STD_ERROR

            return f"The most likely answers are: {', '.join(results_df.Label.values)}"

        except Exception as e:
            logger.info(f"Error during query answering: {e}")
            return STD_ERROR
