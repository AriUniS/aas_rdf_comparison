# filtered_output.py
from typing import Dict, List, Tuple
import requests

from rdf2vec_usage import embed_graphs_from_ttls


'''
ranking.py muss fast nicht geändert werden (nur sicherstellen, dass du die neue embed_graphs nutzt)

Ranking ruft bereits all_embeddings, graph_vectors = embed_graphs(...) auf und macht dann Cosine. 

Das passt. NUR sicherstellen, dass filtered_output_usage.embed_graphs jetzt die neue Version ist.

similarity_measure.py kann so bleiben.'''



def fetch_graph_as_ttl(fuseki_endpoint: str, graph_uri: str) -> str:
    """
    Holt einen Named Graph aus Fuseki als Turtle via SPARQL CONSTRUCT.
    Erwartet, dass fuseki_endpoint auf den SPARQL Endpoint zeigt.
    """
    query = f"""
    CONSTRUCT {{
      ?s ?p ?o .
    }}
    WHERE {{
      GRAPH <{graph_uri}> {{
        ?s ?p ?o .
      }}
    }}
    """

    headers = {"Accept": "text/turtle"}
    r = requests.get(
        fuseki_endpoint,
        params={"query": query},
        headers=headers,
        timeout=60,
    )
    r.raise_for_status()
    return r.text


def embed_graphs(
    fuseki_endpoint: str,
    graph_uris: List[str],
    *,
    vector_size: int = 100,
    max_depth: int = 10,
    max_walks: int = 10,
    epochs: int = 5,
    seed: int = 42,
) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, List[float]]]:
    """
    Lädt alle Graphen (TTL), trainiert EIN gemeinsames Modell und gibt:
      - all_embeddings: per Graph die Node-Embeddings
      - graph_vectors: pro Graph ein gepoolter Graph-Vektor
    zurück.

    Genau das brauchst du für A vs B,C,D,... (vergleichbare Similarities).
    """
    ttl_by_uri: Dict[str, str] = {}
    for gu in graph_uris:
        ttl_by_uri[gu] = fetch_graph_as_ttl(fuseki_endpoint, gu)

    all_embeddings, graph_vectors = embed_graphs_from_ttls(
        ttl_by_uri,
        vector_size=vector_size,
        max_depth=max_depth,
        max_walks=max_walks,
        epochs=epochs,
        seed=seed,
    )
    return all_embeddings, graph_vectors
