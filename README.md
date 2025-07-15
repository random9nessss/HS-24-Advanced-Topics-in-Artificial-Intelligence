# Advanced Topics in Artificial Intelligence (ATAI) @UZH 2024

**Course:** Advanced Topics in Artificial Intelligence  
**University:** University of Zurich (UZH)  
**Year:** 2024  
**Authors:** Sandrin Hunkeler, Kevin Bründler

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Capabilities](#capabilities)  
   - [SPARQL Questions](#sparql-questions)  
   - [Factual Questions](#factual-questions)  
   - [Recommender Questions](#recommender-questions)  
   - [Multimedia Questions](#multimedia-questions)  
   - [Crowd-Sourcing Questions](#crowd-sourcing-questions)  
   - [Small Talk](#small-talk)  
3. [Methods Utilized](#methods-utilized)  
   - [Models Applied](#models-applied)  
   - [Libraries Utilized](#libraries-utilized)  
4. [Examples](#examples)  
5. [Additional Features](#additional-features)  
   - [Query Routing](#query-routing)  
   - [Prefix Tree for Multi-Entity Matching](#prefix-tree-for-multi-entity-matching)  
6. [Conclusions](#conclusions)  
7. [References](#references)

---

## Introduction

Our agent is designed to answer factual questions, recommend movies, and provide comprehensive information about films, actors, and related topics—complete with images. It retrieves detailed knowledge from a knowledge graph using SPARQL queries and can engage in small talk for non-movie inquiries. By leveraging NLP, entity embeddings, and large language models (LLMs), the agent delivers precise and fluent responses. Core functionalities include Named Entity Recognition (NER), fuzzy matching, and embedding-based similarity calculations for contextualized answers.

---

## Capabilities

### SPARQL Questions

Our agent handles SPARQL queries, even if they have partially missing headers. We reconstruct such headers at runtime to fulfill the user’s request.

### Factual Questions

For factual questions, the agent matches entities and filters relevant predicates to gather information. Extracted data from the knowledge graph and embeddings are then used to generate final answers.

**Entity and Predicate Filtering**  
1. **Entity Matching**: Identifies relevant entities by prefix and fuzzy matching.  
2. **Predicate Extraction**: Gathers associated predicates from the knowledge graph and embeddings.  
3. **Predicate Filtering**: Uses embeddings to semantically select predicates matching the user’s context.

**Answer Generation**  
- **Graph-Based Answers**: Collect attributes for matched entities, generate coherent answers with DistilBERT, and refine for conversational fluency using T5-Flan.  
- **Embedding-Based Answers**: Retrieve embeddings for entities and predicates, calculate cosine similarity, and concatenate top matches into a final response.

### Recommender Questions

To recommend movies, we enhanced our fuzzy entity matcher to reliably handle multi-entity queries:

1. **Transform Knowledge Graph** into an undirected NetworkX graph.  
2. **Identify Starting Entities** from user input.  
3. **Entity Augmentation** retrieves genre/director info for deeper matching.  
4. **Random Walks** explore related movies, applying random lengths and a beta attenuation factor.  
5. **Score Movies** for proximity to the starting entity.  
6. **Select Top Recommendations** and present them with metadata.

### Multimedia Questions

We index images by transforming IMDb IDs into knowledge-graph entities. For each entity, we prioritize relevant images (posters for movies, publicity photos for actors) and randomly choose among multiple valid options.

### Crowd-Sourcing Questions

We processed crowd-sourced data by:

1. **Time-Based Filtering**: Removing workers who rushed the questions (<35 seconds each).  
2. **Approval Rate Filtering**: Excluding contributors with approval rates <50%.  
3. **Invalid Answer Filtering**: Discarding irrelevant or improperly formatted responses.

Approved answers are then embedded and made available for retrieval during runtime.

### Small Talk

Our agent can also handle small talk for non-movie queries, powered by FLAN-T5-XL for conversational text generation.

---

## Methods Utilized

### Models Applied

- **Sentence Transformer (all-mpnet-base-v2)**  
  Used for computing sentence embeddings and similarity matching.  

- **DistilBERT (DistilBERT-Base-Uncased-Distilled-SQuAD)**  
  Used for lightweight question-answering with context from the knowledge graph.  

- **FLAN-T5-XL**  
  Used for conversational answer formatting and small talk.  

- **Bart-Large-MNLI**  
  Used for zero-shot classification, determining whether queries relate to movies or other domains.  

- **Fine-tuned DistilBERT**  
  Trained with synthetic data (via GPT-4) to classify user queries into factual, recommendation, multimedia, or small talk categories.

### Libraries Utilized

- **Pandas**  
  Transforms the knowledge graph into a DataFrame, speeding up inference and similarity searches.  

- **NLTK**  
  Filters user queries, removing most stopwords while keeping query-relevant ones like “where,” “when,” etc.  

- **RapidFuzz**  
  Enables robust fuzzy string matching, accommodating user typos or partial matches.  

- **Hugging Face Transformers**  
  Essential for loading and running pretrained NLP models.  

- **NetworkX**  
  Builds an undirected graph (movies, actors, directors, etc.) for random walks and movie recommendations.

---

## Examples

Below are condensed examples of how the agent handles various question types:

1. **Factual Question**  
   - *Question:* “Where was Angelina Jolie born?”  
   - *Routing:* Factual  
   - *Entity Matching:* [Angelina Jolie]  
   - *Answer Generation:* “Angelina Jolie was born in Los Angeles.”

2. **Crowd-Sourcing Question**  
   - *Question:* “What is the box office of The Princess and the Frog?”  
   - *Routing:* Factual  
   - *Crowd Data:* Found a validated answer with partial user agreement.  
   - *Response:* “The box office of The Princess And The Frog is 267000000.”

3. **Small Talk**  
   - *Question:* “Hi, nice to meet you! How is life in the matrix?”  
   - *Routing:* Small Talk  
   - *Response Generation:* “I’m doing great. I’ve been working a lot lately...”

4. **Recommender Question**  
   - *Question:* “I enjoyed The Terminator and anything from Steven Spielberg. What else should I watch?”  
   - *Routing:* Recommender  
   - *Process:* Random walks on the graph, intersection of relevant genres/directors.  
   - *Outcome:* Returns top movie recommendations with metadata (genre, release date, etc.).

5. **Multimedia Question**  
   - *Question:* “What does Angelina Jolie look like?”  
   - *Routing:* Multimedia  
   - *Image Retrieval:* Filters pictures of Angelina Jolie, picks one at random.  
   - *Response:* “Here is an image of Angelina Jolie: [URL/link]”

---

## Additional Features

### Query Routing

We fine-tuned a DistilBERT model using synthetic questions (via GPT-4) to categorize queries into factual, recommendation, multimedia, or small talk. This approach efficiently directs user requests to the correct module.

### Prefix Tree for Multi-Entity Matching

Recognizing multiple entities in a single query posed a challenge for standard NER methods. We implemented a Prefix Tree (Trie) to efficiently match sequences of tokens (e.g., actors’ names, movie titles), allowing approximate matches (Levenshtein distance ≤ 1 for longer tokens).

---

## Conclusions

We developed a versatile agent capable of handling factual questions, multimedia queries, recommendations, and small talk. By integrating structured knowledge graphs, embedding-based reasoning, and advanced NLP models, our system delivers accurate, user-friendly interactions.

Future work may involve:

- **Domain Expansion**: Extending the agent’s functionality to more areas.  
- **Performance Optimization**: Improving real-time performance and reducing latency.  
- **Refined Small Talk**: Offering more varied and context-aware conversations.  

Overall, this project demonstrates the potential of combining structured data, embeddings, and conversational models to build intelligent, domain-specific systems.

*The work was shared equally between both authors.*

---

## References

1. Reimers, N., & Gurevych, I. *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.*  
2. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL, 2019.  
3. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.*  
4. Raffel, C., Shazeer, N., Roberts, A., et al. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5).*  
5. Lewis, M., Liu, Y., Goyal, N., et al. *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.*
