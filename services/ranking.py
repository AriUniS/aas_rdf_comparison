from __future__ import annotations

from typing import Dict, List, Any
import numpy as np

from filtered_output import embed_graphs
from similarity_measure import cosine_similarity


def embed_and_rank_graphs(
    fuseki_endpoint: str,
    graphs: List[str],
    init_graph_uri: str,
) -> Dict[str, Any]:
    """
    1) Erzeuge Embeddings für die gegebenen Graph-URIs (via embed_graphs)
    2) Berechne Cosine-Similarity gegen init_graph_uri (via similarity_measure.cosine_similarity)
    3) Sortiere absteigend und liefere JSON-freundliches Ranking

    Rückgabe:
      {
        "ranking": [{"graph_uri": str, "similarity": float}, ...],
        "count": int
      }
    """
    if not graphs:
        return {"ranking": [], "count": 0}

    if not init_graph_uri:
        raise ValueError("init_graph_uri must be provided for ranking.")

    graphs_to_embed = list(dict.fromkeys([*graphs, init_graph_uri]))  # dedup, Reihenfolge behalten

    # Embeddings holen:
    # Erwartet: embed_graphs(endpoint, graphs) -> Dict[str, vector]
    #embeddings: Dict[str, Any] = embed_graphs(fuseki_endpoint, graphs)


    all_embeddings, graph_vectors = embed_graphs(fuseki_endpoint, graphs_to_embed)

    #embeddings: Dict[str, Any] = embed_graphs(fuseki_endpoint, graphs_to_embed)
    #print(graph_vectors)


    if init_graph_uri not in graph_vectors:
        raise ValueError(
            f"init_graph_uri '{init_graph_uri}' has no embedding. "
            "Make sure init_graph_uri is included in 'graphs' or can be embedded."
        )

    init_vec = graph_vectors[init_graph_uri]

    ranking: List[Dict[str, float | str]] = []
    for graph_uri, vec in graph_vectors.items():
        if graph_uri == init_graph_uri:
            continue

        sim = cosine_similarity(init_vec, vec)
        ranking.append({"graph_uri": graph_uri, "similarity": float(sim)})

    ranking.sort(key=lambda x: x["similarity"], reverse=True)

    return {"ranking": ranking, "count": len(ranking)}